#include "md_controller/com.hpp"
#include <string>

serial::Serial ser;

#ifndef PID_VEL_CMD_DUAL
#define PID_VEL_CMD_DUAL 207  // 0xCF, dual-channel velocity command
#endif

// ----- 유틸 -----

// Get the low and high byte from short
IByte Short2Byte(short sIn)
{
    IByte Ret;
    Ret.byLow  = (BYTE)(sIn & 0xff);
    Ret.byHigh = (BYTE)((sIn >> 8) & 0xff);
    return Ret;
}

// Make short data from two bytes (little endian)
int Byte2Short(BYTE byLow, BYTE byHigh)
{
    return (int)byLow | ((int)byHigh << 8);
}

// Make long data from four bytes (little endian)
int Byte2LInt(BYTE byData1, BYTE byData2, BYTE byData3, BYTE byData4)
{
    return (int)byData1 | ((int)byData2 << 8) | ((int)byData3 << 16) | ((int)byData4 << 24);
}

// ----- 시리얼 초기화 -----

int InitSerial(void)
{
    try
    {
        ser.setPort(Com.nPort);
        ser.setBaudrate(Com.nBaudrate);
        // 57600 기준 ~1.6ms, 115200 기준 ~0.35ms에 맞춰 사용하던 값 유지
        serial::Timeout to = serial::Timeout::simpleTimeout(1667);
        ser.setTimeout(to);
        ser.open();
    }
    catch (serial::IOException& e)
    {
        RCLCPP_ERROR_STREAM(rclcpp::get_logger("rclcpp"), "Unable to open port ");
        return -1;
    }

    if (ser.isOpen())
        RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "Serial Port initialized");
    else
        return -1;

    return 0;
}

// ----- 송신 프레임 생성/전송 -----

// for sending the data (One ID)
// byMID: 보낸 쪽(PC/호스트) 또는 상위 주소로 기존 코드에서 사용하던 값
// id_num: 장치(드라이버) ID
int PutMdData(BYTE byPID, BYTE byMID, int id_num, int nArray[])
{
    IByte iData;
    BYTE byPidDataSize, byDataSize, i, j;
    static BYTE byTempDataSum;

    for (j = 0; j < MAX_PACKET_SIZE; j++) Com.bySndBuf[j] = 0;

    // 기존 코드 포맷 유지
    // [0]=byMID, [1]=0xB8, [2]=id_num, [3]=byPID, [4]=data_len, [5..]=data, [last]=checksum(two's complement)
    Com.bySndBuf[0] = byMID;
    Com.bySndBuf[1] = 184;      // 0xB8
    Com.bySndBuf[2] = (BYTE)id_num;
    Com.bySndBuf[3] = byPID;

    switch (byPID)
    {
        case PID_REQ_PID_DATA:
        {
            byDataSize    = 1;
            byPidDataSize = 7;   // 5(header+len) + 1(data) + 1(chksum)
            byTempDataSum = 0;

            Com.bySndBuf[4] = byDataSize;
            Com.bySndBuf[5] = (BYTE)nArray[0];

            for (i = 0; i < (byPidDataSize - 1); i++) byTempDataSum += Com.bySndBuf[i];
            Com.bySndBuf[byPidDataSize - 1] = (BYTE)(~byTempDataSum + 1);

            ser.write(Com.bySndBuf, byPidDataSize);
            break;
        }

        case PID_POSI_RESET:
        {
            byDataSize    = 1;
            byPidDataSize = 7;
            byTempDataSum = 0;

            Com.bySndBuf[4] = byDataSize;
            Com.bySndBuf[5] = (BYTE)nArray[0];

            for (i = 0; i < (byPidDataSize - 1); i++) byTempDataSum += Com.bySndBuf[i];
            Com.bySndBuf[byPidDataSize - 1] = (BYTE)(~byTempDataSum + 1);

            ser.write(Com.bySndBuf, byPidDataSize);
            break;
        }

        case PID_COMMAND:
        {
            byDataSize    = 1;
            byPidDataSize = 7;
            byTempDataSum = 0;

            Com.bySndBuf[4] = byDataSize;
            Com.bySndBuf[5] = (BYTE)nArray[0];

            for (i = 0; i < (byPidDataSize - 1); i++) byTempDataSum += Com.bySndBuf[i];
            Com.bySndBuf[byPidDataSize - 1] = (BYTE)(~byTempDataSum + 1);

            ser.write(Com.bySndBuf, byPidDataSize);
            break;
        }

        case PID_VEL_CMD:
        {
            // 기존 단일 채널 속도: data=2바이트
            byDataSize    = 2;
            byPidDataSize = 8;   // 5 + 2 + 1
            byTempDataSum = 0;

            Com.bySndBuf[4] = byDataSize;
            Com.bySndBuf[5] = (BYTE)nArray[0];
            Com.bySndBuf[6] = (BYTE)nArray[1];

            for (i = 0; i < (byPidDataSize - 1); i++) byTempDataSum += Com.bySndBuf[i];
            Com.bySndBuf[byPidDataSize - 1] = (BYTE)(~byTempDataSum + 1);

            ser.write(Com.bySndBuf, byPidDataSize);
            // ser.flush(); // 필요시 사용
            break;
        }

        case PID_VEL_CMD_DUAL:
        {
            // 듀얼 속도(0xCF): data 7바이트 [01 LL LH 01 RL RH 00]
            byDataSize    = 7;
            byPidDataSize = 13;  // 5 + 7 + 1
            byTempDataSum = 0;

            Com.bySndBuf[4]  = byDataSize;
            Com.bySndBuf[5]  = (BYTE)nArray[0];
            Com.bySndBuf[6]  = (BYTE)nArray[1];
            Com.bySndBuf[7]  = (BYTE)nArray[2];
            Com.bySndBuf[8]  = (BYTE)nArray[3];
            Com.bySndBuf[9]  = (BYTE)nArray[4];
            Com.bySndBuf[10] = (BYTE)nArray[5];
            Com.bySndBuf[11] = (BYTE)nArray[6];

            for (i = 0; i < (byPidDataSize - 1); i++) byTempDataSum += Com.bySndBuf[i];
            Com.bySndBuf[byPidDataSize - 1] = (BYTE)(~byTempDataSum + 1);

            ser.write(Com.bySndBuf, byPidDataSize);
            ser.flush();

            // 다음 프레임 유실 방지용: 입력 버퍼 드레인
            if (ser.available()) {
                BYTE dump[64];
                size_t avail = ser.available();
                size_t toread = (avail < 64) ? avail : 64;
                ser.read(dump, toread);
            }
            break;
        }
    }

    return SUCCESS;
}

// ----- 듀얼/채널별 래퍼 -----

// 내부 저장: 마지막 좌/우 명령값 (단일 채널 갱신 시 반대쪽 유지)
static short __last_left  = 0;
static short __last_right = 0;

static inline short clamp16(short v)
{
    if (v >  32767) return  32767;
    if (v < -32767) return -32767;
    return v;
}

// 한 프레임(0xCF)으로 좌/우 동시 제어
int SetDualRpm(short rpm_left, short rpm_right)
{
    rpm_left  = clamp16(rpm_left);
    rpm_right = clamp16(rpm_right);

    IByte L = Short2Byte(rpm_left);
    IByte R = Short2Byte(rpm_right);

    int a[7];
    a[0] = 0x01;        // CH1 flag
    a[1] = L.byLow;
    a[2] = L.byHigh;
    a[3] = 0x01;        // CH2 flag
    a[4] = R.byLow;
    a[5] = R.byHigh;
    a[6] = 0x00;        // reserved

    // 기존 호출 패턴과 동일하게: byMID=Com.nRMID, id_num=Motor.ID 사용 (필요 시 프로젝트에 맞게 조정)
    BYTE byMID  = (BYTE)Com.nIDMDT;
    int  id_num = Motor.ID;

    __last_left  = rpm_left;
    __last_right = rpm_right;

    return PutMdData(PID_VEL_CMD_DUAL, byMID, id_num, a);
}

int SetRpmLeft(short rpm)
{
    __last_left = clamp16(rpm);
    return SetDualRpm(__last_left, __last_right);
}

int SetRpmRight(short rpm)
{
    __last_right = clamp16(rpm);
    return SetDualRpm(__last_left, __last_right);
}

int SetRpmChannel(BYTE ch, short rpm)
{
    if (ch == 1) return SetRpmLeft(rpm);
    if (ch == 2) return SetRpmRight(rpm);
    return FAIL;
}

// ----- 수신 처리 -----

int MdReceiveProc(void) // save the identified serial data to defined variable according to PID NUMBER data
{
    BYTE byRcvRMID, byRcvTMID, byRcvID, byRcvPID, byRcvDataSize;

    byRcvRMID     = Com.byRcvBuf[0];
    byRcvTMID     = Com.byRcvBuf[1];
    byRcvID       = Com.byRcvBuf[2];
    byRcvPID      = Com.byRcvBuf[3];
    byRcvDataSize = Com.byRcvBuf[4];

    switch (byRcvPID)
    {
        case PID_MAIN_DATA:
            Com.rpm      = Byte2Short(Com.byRcvBuf[5], Com.byRcvBuf[6]);
            Com.position = Byte2LInt(Com.byRcvBuf[15], Com.byRcvBuf[16], Com.byRcvBuf[17], Com.byRcvBuf[18]);
            break;
    }

    return SUCCESS;
}

int AnalyzeReceivedData(BYTE byArray[], BYTE byBufNum) // Analyze the communication data
{
    static BYTE byChkSec;
    BYTE i, j;
    int count = 0;

    if (Com.byPacketNum >= MAX_PACKET_SIZE)
    {
        Com.byStep = 0;
        return FAIL;
    }

    for (j = 0; j < byBufNum; j++)
    {
        switch (Com.byStep)
        {
            case 0: // header: 0xB8 or 0xB7
                if ((byArray[j] == 184) || (byArray[j] == 183))
                {
                    Com.byChkSum += byArray[j];
                    Com.byRcvBuf[Com.byPacketNum++] = byArray[j];
                    Com.byChkComError = 0;
                    count++;
                    if (count == 2) Com.byStep++;
                }
                else
                {
                    // printf("ERROR (1)\n");
                    count = 0;
                    Com.byStep      = 0;
                    Com.fgPacketOK  = 0;
                    Com.byPacketNum = 0;
                    Com.byChkComError++;
                }
                break;

            case 1: // ID
                if (byArray[j] == 1 || byArray[j] == 2)
                {
                    Com.byChkSum += byArray[j];
                    Com.byRcvBuf[Com.byPacketNum++] = byArray[j];
                    Com.byStep++;
                    Com.byChkComError = 0;
                }
                else
                {
                    // printf("ERROR (2)\n");
                    Com.byStep = 0;
                    Com.byPacketNum = 0;
                    Com.byChkComError++;
                }
                break;

            case 2: // PID
                Com.byChkSum += byArray[j];
                Com.byRcvBuf[Com.byPacketNum++] = byArray[j];
                Com.byStep++;
                break;

            case 3: // data length
                Com.byMaxDataNum = byArray[j];
                Com.byDataNum = 0;
                Com.byChkSum += byArray[j];
                Com.byRcvBuf[Com.byPacketNum++] = byArray[j];
                Com.byStep++;
                break;

            case 4: // data
                Com.byRcvBuf[Com.byPacketNum++] = byArray[j];
                Com.byChkSum += byArray[j];

                if (++Com.byDataNum >= MAX_DATA_SIZE)
                {
                    // printf("check 5\n");
                    Com.byStep = 0;
                    Com.byTotalRcvDataNum = 0;
                    break;
                }

                if (Com.byDataNum >= Com.byMaxDataNum) Com.byStep++;
                break;

            case 5: // checksum
                Com.byChkSum += byArray[j];
                Com.byRcvBuf[Com.byPacketNum++] = byArray[j];

                if (Com.byChkSum == 0)
                {
                    Com.fgPacketOK   = 1;
                    Com.fgComDataChk = 1;
                    Com.byDataNum    = 0;
                    Com.byMaxDataNum = 0;
                }

                Com.byStep = 0;
                Com.byTotalRcvDataNum = 0;
                break;

            default:
                Com.byStep = 0;
                Com.fgComComple = ON;
                break;
        }

        if (Com.fgPacketOK)
        {
            Com.fgPacketOK   = 0;
            Com.byPacketSize = 0;
            Com.byPacketNum  = 0;

            if (byChkSec == 0) byChkSec = 1;

            MdReceiveProc(); // save the identified serial data
        }

        if (Com.byChkComError == 10) // while 50ms
        {
            // printf("check error\n");
            Com.byChkComError = 0;
            Com.byStep = 0;
            Com.byChkSum = 0;
            Com.byMaxDataNum = 0;
            Com.byDataNum = 0;
            for (i = 0; i < MAX_PACKET_SIZE; i++) Com.byRcvBuf[i] = 0;
            j = byBufNum;
        }
    }
    return SUCCESS;
}

int ReceiveDataFromController(BYTE init) // read & analyze
{
    BYTE byRcvBuf[250];
    BYTE byBufNumber;

    byBufNumber = ser.available();

    if (byBufNumber != 0)
    {
        byBufNumber = MAX_DATA_SIZE; // 기존 로직 유지
        ser.read(byRcvBuf, byBufNumber);

        if (init == ON)
        {
            if (byRcvBuf[2] == Motor.ID)
            {
                RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "ID %d Motor Init success!", Motor.ID);
                Motor.InitMotor = OFF;
            }
        }
        else
        {
            AnalyzeReceivedData(byRcvBuf, byBufNumber);
        }
    }
    return 1;
}
