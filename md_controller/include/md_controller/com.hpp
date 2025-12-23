#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <iostream>
#include <float.h>
#include <math.h>

#include "rclcpp/rclcpp.hpp"
#include "serial/serial.h"
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/bool.hpp>
#include "std_msgs/msg/int16_multi_array.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include <tf2/LinearMath/Quaternion.h>

using namespace std;

#define _X                  0
#define _Y                  1
#define _THETA              2
#define PI                  3.14159265359

#define ON                  1
#define OFF                 0
#define RESET               0
#define FAIL                0
#define SUCCESS             1

#define Abs(a)              (((a)<(0)) ? -(a):(a))

// ===== MD 프로토콜 PID =====
#define PID_REQ_PID_DATA    4
#define PID_TQ_OFF          5
#define PID_COMMAND         10
#define PID_POSI_RESET      13
#define PID_VEL_CMD         130          // (단일 채널용일 수 있음, 기존 코드에서 사용)
#define PID_MAIN_DATA       193

// ★ 추가: 양 채널 속도 명령 (장비 로그 기준 0xCF = 207)
#define PID_VEL_CMD_DUAL    207          // 0xCF

// (선택) 채널 식별자
#define CH1                 1
#define CH2                 2

#define MAX_PACKET_SIZE     26
#define MAX_DATA_SIZE       23

#define REQUEST_PNT_MAIN_DATA 2

#define DURATION            0.0001

#define TIME_50MS           1
#define TIME_100MS          2
#define TIME_1S             20
#define TIME_5S             100

typedef unsigned char  BYTE;
typedef unsigned short WORD;
typedef unsigned int   DWORD;

typedef struct {
    BYTE bySndBuf[MAX_PACKET_SIZE];
    BYTE byRcvBuf[MAX_PACKET_SIZE];
    BYTE byPacketSize;
    BYTE byPacketNum;
    BYTE byIn, byStep;
    BYTE byChkSend;
    BYTE byChkRcv;
    BYTE fgInIdleLine, fgPacketOK, fgComComple;
    BYTE byTotalRcvDataNum;
    BYTE fgChk;
    BYTE byChkSum, byMaxDataNum, byDataNum;

    string nPort;
    int nIDPC, nIDMDUI, nIDMDT, nRMID;
    int nBaudrate, nWheelLength, fgDirSign;
    short sSetDia, sSetWheelLen, sSetGear;
    int nCmdSpeed, nCmdAngSpeed;
    int nSlowstart, nSlowdown;
    float nWheelDiameter;
    int rpm,position;

    BYTE byChkComError;
    BYTE fgComDataChk;
    BYTE fgInitsetting;

} Communication;
extern Communication Com;

typedef struct{
    int ID, GearRatio, InitError, poles;
    BYTE InitMotor;
    short rpm;
    long position;
    float current_tick, last_diff_tick, last_tick, last_rad, Tick2RAD, PPR;
} MotorVar;
extern MotorVar Motor;

typedef struct {
    BYTE byLow;
    BYTE byHigh;
} IByte;

extern IByte Short2Byte(short sIn);
extern int Byte2Short(BYTE byLow, BYTE byHigh);
extern int Byte2LInt(BYTE byData1, BYTE byData2, BYTE byData3, BYTE byData4);

extern int InitSerial(void);
extern int InitSetParam(void);
extern int PutMdData(BYTE byPID, BYTE byID, int id_num, int nArray[]);
extern int MdReceiveProc(void);
extern int ReceiveDataFromController(BYTE init);
extern int AnalyzeReceivedData(BYTE byArray[], BYTE byBufNum);

// ===== ★ 추가: 듀얼 모터 제어용 API =====
// - 두 채널을 한 프레임(0xCF)으로 동시 제어
extern int SetDualRpm(short rpm_left, short rpm_right);

// (옵션) 단일 채널 제어가 필요하면 아래도 제공
extern int SetRpmChannel(BYTE ch, short rpm);   // ch=CH1 또는 CH2
extern int SetRpmLeft(short rpm);               // SetRpmChannel(CH1, rpm) 래퍼
extern int SetRpmRight(short rpm);              // SetRpmChannel(CH2, rpm) 래퍼
