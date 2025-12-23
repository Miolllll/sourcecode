#include "md_controller/com.hpp"

Communication Com;
MotorVar Motor;

geometry_msgs::msg::TransformStamped odom_tf;
sensor_msgs::msg::JointState joint_states;

// ==== 추가: 좌/우 명령값 저장 ====
int rpm_left_  = 0;
int rpm_right_ = 0;

// 기존 단일 토픽(/cmd_rpm) 호환: 같은 값으로 좌/우 모두 설정
void CmdRpmCallBack(const std_msgs::msg::Int32::SharedPtr msg) {
    rpm_left_  = msg->data;
    rpm_right_ = msg->data;
}

// 좌측 전용
void CmdRpmLeftCallBack(const std_msgs::msg::Int32::SharedPtr msg) {
    rpm_left_ = msg->data;
}

// 우측 전용
void CmdRpmRightCallBack(const std_msgs::msg::Int32::SharedPtr msg) {
    rpm_right_ = msg->data;
}

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("md_controller_node");

    // create TF
    rclcpp::Time stamp_now;
    tf2_ros::TransformBroadcaster tf_broadcaster_(node);

    // ==== 구독자 ====
    // 1) 단일 토픽(호환)
    auto rpm_sub_both = node->create_subscription<std_msgs::msg::Int32>(
        "/cmd_rpm", 10, CmdRpmCallBack);
    // 2) 좌/우 개별
    auto rpm_sub_left = node->create_subscription<std_msgs::msg::Int32>(
        "/cmd_rpm_left", 10, CmdRpmLeftCallBack);
    auto rpm_sub_right = node->create_subscription<std_msgs::msg::Int32>(
        "/cmd_rpm_right", 10, CmdRpmRightCallBack);

    // ==== 파라미터 ====
    node->declare_parameter("MDUI", 184);
    node->declare_parameter("MDT", 183);
    node->declare_parameter("Port", "/dev/ttyUSB0");
    node->declare_parameter("Baudrate", 19200);
    node->declare_parameter("ID", 1);
    node->declare_parameter("GearRatio", 20);
    node->declare_parameter("poles", 4);

    // (옵션) 채널 방향 반전 플래그
    node->declare_parameter("invert_left",  true);
    node->declare_parameter("invert_right", true);

    node->get_parameter("MDUI", Com.nIDMDUI);
    node->get_parameter("MDT", Com.nIDMDT);
    node->get_parameter("Port", Com.nPort);
    node->get_parameter("Baudrate", Com.nBaudrate);
    node->get_parameter("ID", Motor.ID);
    node->get_parameter("GearRatio", Motor.GearRatio);
    node->get_parameter("poles", Motor.poles);

    bool invert_left  = node->get_parameter("invert_left").as_bool();
    bool invert_right = node->get_parameter("invert_right").as_bool();

    Motor.PPR       = Motor.poles * 3 * Motor.GearRatio;   // poles * 3(HALL U,V,W) * gear ratio
    Motor.Tick2RAD  = (360.0 / Motor.PPR) * PI / 180.0;

    IByte iData;
    int nArray[2];
    static BYTE fgInitsetting, byCntInitStep, byCntComStep, byCnt2500us, byCntStartDelay, byCntCase[5];

    byCntInitStep     = 1;
    Motor.InitMotor   = ON;
    fgInitsetting     = OFF;
    Motor.InitError   = 0;
    Motor.last_rad    = 0;
    Motor.last_tick   = 0;

    // 통신 초기화
    InitSerial();

    while (rclcpp::ok()) {

        // 수신 처리 (초기에는 ID 확인용)
        ReceiveDataFromController(Motor.InitMotor);

        // 주기 카운터 (원 코드 유지) - 2.5ms tick 가정, 50회 -> 100ms
        if (++byCnt2500us == 50) {
            byCnt2500us = 0;

            if (fgInitsetting == ON) {
                switch (++byCntComStep) {
                case 1: { // TF & 위치 업데이트
                    geometry_msgs::msg::TransformStamped transformStamped;
                    transformStamped.header.stamp = node->now();
                    transformStamped.header.frame_id = "world";
                    transformStamped.child_frame_id = "motor_joint";

                    transformStamped.transform.translation.x = 0.0;
                    transformStamped.transform.translation.y = 0.0;
                    transformStamped.transform.translation.z = 0.15;

                    Motor.current_tick     = Com.position;

                    Motor.last_diff_tick   = Motor.current_tick - Motor.last_tick;
                    Motor.last_tick        = Motor.current_tick;
                    Motor.last_rad        += Motor.Tick2RAD * (double)Motor.last_diff_tick;

                    tf2::Quaternion q;
                    q.setRPY(0, 0, -Motor.last_rad);
                    transformStamped.transform.rotation.x = q.x();
                    transformStamped.transform.rotation.y = q.y();
                    transformStamped.transform.rotation.z = q.z();
                    transformStamped.transform.rotation.w = q.w();

                    tf_broadcaster_.sendTransform(transformStamped);
                    break;
                }
                case 2: // 제어 + 상태 요청
                    if (++byCntCase[byCntComStep] == TIME_100MS) {
                        byCntCase[byCntComStep] = 0;

                        // ==== 여기서 듀얼 속도 프레임(0xCF) 전송 ====
                        // 기존 코드가 motor rpm = wheel rpm * gear ratio 로 쓰고 있으므로 동일 적용
                        int left_cmd  = rpm_left_  * Motor.GearRatio;
                        int right_cmd = rpm_right_ * Motor.GearRatio;

                        if (invert_left)  left_cmd  = -left_cmd;
                        if (invert_right) right_cmd = -right_cmd;

                        // 0xCF 듀얼 속도 전송 (com.cpp의 SetDualRpm 사용)
                        SetDualRpm((short)left_cmd, (short)right_cmd);

                        // 메인 데이터 요청 (원 코드 유지)
                        nArray[0] = PID_MAIN_DATA;
                        PutMdData(PID_REQ_PID_DATA, Com.nIDMDT, Motor.ID, nArray);

                    }
                    byCntComStep = 0;
                    break;
                }
            } else {
                // === 초기화 단계 ===
                if (byCntStartDelay <= 200) byCntStartDelay++;
                else {
                    switch (byCntInitStep)
                    {
                    case 1: // Motor connect check
                        nArray[0] = PID_MAIN_DATA;
                        PutMdData(PID_REQ_PID_DATA, Com.nIDMDT, Motor.ID, nArray);

                        if (Motor.InitMotor == ON)
                            Motor.InitError++;
                        else
                            byCntInitStep++;

                        if (Motor.InitError > 10) {
                            RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "ID %d MOTOR INIT ERROR!!", Motor.ID);
                            return 0;
                        }
                        break;

                    case 2:
                        byCntInitStep++;
                        break;

                    case 3: // Motor torque ON (속도 0 전송)
                        nArray[0] = 0;
                        nArray[1] = 0;
                        PutMdData(PID_VEL_CMD, Com.nIDMDT, Motor.ID, nArray);
                        byCntInitStep++;
                        break;

                    case 4: // Motor POS reset
                        nArray[0] = 0;
                        PutMdData(PID_POSI_RESET, Com.nIDMDT, Motor.ID, nArray);
                        byCntInitStep++;
                        break;

                    case 5:
                        printf("========================================================\n\n");
                        RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "MOTOR INIT END\n");
                        fgInitsetting = ON;
                        break;
                    }
                }
            }
        }

        rclcpp::spin_some(node);
    }

    rclcpp::shutdown();
    return 0;
}
