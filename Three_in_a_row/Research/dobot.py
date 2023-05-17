import threading
import DobotDllType as dType

# Define connection status messages
CON_STR = {
    dType.DobotConnect.DobotConnect_NoError: "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied",
}

# Load Dll
api = dType.load()

# Connect Dobot
state = dType.ConnectDobot(api, "", 115200)[0]
print("Connect status:", CON_STR[state])


if state == dType.DobotConnect.DobotConnect_NoError:

    # Clean Command Queued
    dType.SetQueuedCmdClear(api)

    # Async Motion Params Setting
    dType.SetHOMEParams(api, 200, 200, 200, 200, isQueued=0)
    dType.SetPTPJointParams(api, 200, 200, 200, 200, 200, 200, 200, 200, isQueued=0)
    dType.SetPTPCommonParams(api, 100, 100, isQueued=0)

    # Async Home
    dType.SetHOMECmd(api, temp=0, isQueued=0)
    dType.SetQueuedCmdStartExec(api)

    dType.SetQueuedCmdClear(api)
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, 253.32440185546875, -112.41374206542969, -39.64093017578125, 0, isQueued=1)
    dType.dSleep(1)
    dType.SetEndEffectorSuctionCup(api, 1, 1)
    dType.dSleep(1)
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, 176.35801696777344, -175.05938720703125, 53.13313293457031, 0, isQueued=1)
    dType.dSleep(1)
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, 84.12555694580078, -269.9014892578125, -15.043380737304688, 0, isQueued=1)
    dType.dSleep(1)
    dType.SetEndEffectorSuctionCup(api, 0, 1)
    dType.SetQueuedCmdStartExec(api)

    # Get current position
    pose = dType.GetPose(api)
    print("Current position:", pose)

# Disconnect Dobot
dType.DisconnectDobot(api)



    
