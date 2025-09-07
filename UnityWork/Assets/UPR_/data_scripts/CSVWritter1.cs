using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Text;
using System.IO;
using System;
using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.Utilities;


public class CSVWritter1 : MonoBehaviour
{
    private StreamWriter writer;
    private float defaultDistanceInMeters = 2f;
    private Vector3 GazeHitPoint, GazeDirection, GazeOrigin, PlayerPos;
    private Quaternion PlayerRot;
    public bool ExpStarted = false;
    private bool GrippedForInsetr = false;
    private int markerX = 0;

    //new variables
    [SerializeField] private GameObject Player;
    private GameObject[] BigTargets, BigTargetsSequenced = new GameObject[3];
    private float[] Yvalue = new float[3];
    //private GameObject[] StatusFlag = new GameObject[3];
    private GameObject[] StatusFlag = new GameObject[2];  //change from 3 to 2 due to floor reduce
    private Transform Status0, Status1, Status2;
    private bool allset = false;
    private bool[] targetStatus = new bool[3];

    //[HideInInspector] private MapSelecter MapSelected;
    private GameObject Map_Sel_Obj;

    [Tooltip("00 for physical , 01 for virtual")] 
    public string ConditionNum;
    public string SerNum;
    private GameObject STCamera;
    private bool SeeThroughActive = false;

    //new new data recording
    public GameObject LuminSensor, PupilRecorder, DataRecorder, LumCam, TargeObjecttCube, CapsuleBase, RobotEndEffector;

    private float LightSensorValue, PupilDiameter, PupilID, CogLoad, Trend, TrendSum, PupilAvg;
    private double LightCalculateValue;
    private Transform RHindexTip, RHthumbTip, RHpalmCenter, RedCube, CapsuleRotated, RobotGripper;

    private IMixedRealityHandJointService handJointService;

    //Collision Detection
    public CollisionDetection collisionDetector;
    int currentCollisionCountA, currentCollisionCountB, currentCollisionCountC, currentCollisionCountD;

    //Grabbing insertion condition flag
    public SnapToHand0 IGConditionSelector;
    private string ConditionIxGx;
    public GameObject GrabTarget, InsertionTarget;
    private Transform GrabTargetTrans, InsertionTargetTrans;

    //switching conditions
    [Tooltip("input conditions: '00'' or '01''")]
    string Condition;

    //Get the Start Status
    public GameObject objectB;
    private bool StartingPosition; 


    void Start()
    {
        InitializeCSV();
        handJointService = CoreServices.GetInputSystemDataProvider<IMixedRealityHandJointService>();
        GrabTargetTrans = GrabTarget.transform;
        InsertionTargetTrans = InsertionTarget.transform;
    }

    // Update is called once per frame
    void Update()
    {
        WriteCSV();
    }


    void OnApplicationQuit()
    {
        FinishCSV();
    }


    void InitializeCSV()
    {
        string filePath = "Assets/Data/" + ConditionNum + "_" + SerNum + ".csv";
        if (File.Exists(filePath))
        {
            Debug.Log("Data file already exists! Make sure to delete or change : )");
            Application.Quit();
        }
        
        writer = new StreamWriter(filePath);
        
        writer.WriteLine("SerialNumber,WorldTime,WorldTime(seconds),WorldTime(MiliSecond),Time.time,FrameNum,ExpStarted,GrippedForInsert," +
            "PlayerX,PlayerY,PlayerZ,PlayerRx,PlayerRy,PlayerRz,PlayerRw," +
            "Lum_Sensor,Lum_RGB,PupilDiameter,PupilID,CogLoad,markerX," +
            "RHindexTip_X,RHindexTip_Y,RHindexTip_Z,RHindexTip_rx,RHindexTip_ry,RHindexTip_rz,RHindexTip_rw," +
            "RHthumbTip_X,RHthumbTip_Y,RHthumbTip_Z,RHthumbTip_rx,RHthumbTip_ry,RHthumbTip_rz,RHthumbTip_rw," +
            "RHpalmCenter_X,RHpalmCenter_Y,RHpalmCenter_Z,RHpalmCenter_rx,RHpalmCenter_ry,RHpalmCenter_rz,RHpalmCenter_rw," +
            "RedCube_X, RedCube_Y, RedCube_Z,RedCube_rx,RedCube_ry,RedCube_rz,RedCube_rw," +
            "CapsuleRotated_X,CapsuleRotated_Y,CapsuleRotated_Z,CapsuleRotated_rx,CapsuleRotated_ry,CapsuleRotated_rz,CapsuleRotated_rw," +
            "RobotGripper_X,RobotGripper_Y,RobotGripper_Z,RobotGripper_rx,RobotGripper_ry,RobotGripper_rz,RobotGripper_rw," +
            "CollisionA,CollisionB,CollisionC,CollisionD,ConditionIxGx," +
            "GrabTargetTransX,GrabTargetTransY,GrabTargetTransZ," +
            "InsertionTargetTransX,InsertionTargetTransY,InsertionTargetTransZ,"+
            "InsideGrabArea,StartingPosition"); //INCLUDES ONLY GPT OUTPUT

    }

    void FinishCSV()
    {
        writer.Flush();
        writer.Close();
    }

    void WriteCSV()
    {
        string fileName = ConditionNum + "_" + SerNum;

        PlayerPos = Player.transform.position;
        PlayerRot = Player.transform.rotation;

        LightSensorValue = LuminSensor.GetComponent<SerialLuminanceReader>().luminanceValue;
        LightCalculateValue = LumCam.GetComponent<Luminosity>().lum;
        PupilDiameter = PupilRecorder.GetComponent<PupilLabs.Demos.PupilDataDemo>().pupil_size;
        PupilID = PupilRecorder.GetComponent<PupilLabs.Demos.PupilDataDemo>().pupil_ID;
        CogLoad = DataRecorder.GetComponent<CogLoad>().getNewCog();
        //Trend = CogLoadMonitor.GetComponent<CogLoad>().getTrend();
        //TrendSum = GetComponent<CogLoad>().getTrendsum();
        //PupilAvg = GetComponent<CogLoad>().getPupilAvg();

        //hand tracking
        RHindexTip = handJointService.RequestJointTransform(TrackedHandJoint.IndexTip, Handedness.Right);
        RHthumbTip = handJointService.RequestJointTransform(TrackedHandJoint.ThumbTip, Handedness.Right);
        RHpalmCenter = handJointService.RequestJointTransform(TrackedHandJoint.Palm, Handedness.Right);
        //redcube for robot control tracking
        RedCube = TargeObjecttCube.transform;
        CapsuleRotated = CapsuleBase.transform;
        RobotGripper = RobotEndEffector.transform;

        //Collision Detection
        currentCollisionCountA = collisionDetector.collisionCountA;
        currentCollisionCountB = collisionDetector.collisionCountB;
        currentCollisionCountC = collisionDetector.collisionCountC;
        currentCollisionCountD = collisionDetector.collisionCountD;

        //gripping insertion condition flag
        ConditionIxGx = IGConditionSelector.ConditionIG;

        //Starging Position Check
        StartingPosition = objectB.activeSelf;

        //set manual event input as additional marker
        if (Input.GetKey(KeyCode.X))
        {
            markerX = 1;
        }
        else 
        {
            markerX = 0;
        }

        if (Input.GetKey(KeyCode.S))
        {
            ExpStarted = true;
        }

        if (Input.GetKey(KeyCode.E))
        {
            ExpStarted = false;
        }

        //press I to mark the startng of the experiment
        if (Input.GetKeyDown(KeyCode.S))
        {
            Debug.Log("Experiment Started");
            ExpStarted = true;
        }
     
        if (Input.GetKeyDown(KeyCode.E)) {
            Debug.Log("Experiment Stopped");
            ExpStarted = false;
        }

        //mark the experiment state to gripped and insert
        if (Input.GetKeyDown(KeyCode.G))
        {
            Debug.Log("Gripped");
            GrippedForInsetr = true;
        }

        writer.WriteLine(fileName + "," + System.DateTime.Now + "," + System.DateTime.Now.Second + "," + System.DateTime.Now.Millisecond + "," + 
            Time.time + "," + Time.frameCount + "," + ExpStarted + "," + GrippedForInsetr + "," +
            PlayerPos.x + "," + PlayerPos.y + "," + PlayerPos.z + "," + PlayerRot.x + "," + PlayerRot.y + "," + PlayerRot.z + "," + PlayerRot.w + "," + 
            LightSensorValue + "," + LightCalculateValue + "," + PupilDiameter + "," + PupilID + "," + CogLoad + "," + markerX + "," + 
            RHindexTip.position.x + "," + RHindexTip.position.y + "," + RHindexTip.position.z + "," + RHindexTip.rotation.x + "," + RHindexTip.rotation.y + "," + RHindexTip.rotation.z + "," + RHindexTip.rotation.w + "," + 
            RHthumbTip.position.x +"," + RHthumbTip.position.y + "," + RHthumbTip.position.z + "," + RHthumbTip.rotation.x + "," + RHthumbTip.rotation.y + "," + RHthumbTip.rotation.z + "," + RHthumbTip.rotation.w + "," + 
            RHpalmCenter.position.x+"," + RHpalmCenter.position.y + "," + RHpalmCenter.position.z + "," + RHpalmCenter.rotation.x + "," + RHpalmCenter.rotation.y + "," + RHpalmCenter.rotation.z + "," + RHpalmCenter.rotation.w +"," +
            RedCube.position.x+","+ RedCube.position.y + "," + RedCube.position.z + "," + RedCube.rotation.x + "," + RedCube.rotation.y + "," + RedCube.rotation.z + "," + RedCube.rotation.w + "," + 
            CapsuleRotated.position.x +","+ CapsuleRotated.position.y + "," + CapsuleRotated.position.z + "," + CapsuleRotated.rotation.x + "," + CapsuleRotated.rotation.y + "," + CapsuleRotated.rotation.z + "," + CapsuleRotated.rotation.w + "," + 
            RobotGripper.position.x +","+ RobotGripper.position.y + "," + RobotGripper.position.z + "," + RobotGripper.rotation.x + "," + RobotGripper.rotation.y + "," + RobotGripper.rotation.z + "," + RobotGripper.rotation.w + "," +
            currentCollisionCountA +","+ currentCollisionCountB+","+ currentCollisionCountC+","+ currentCollisionCountD+","+ ConditionIxGx+","+ 
            GrabTargetTrans.position.x+","+ GrabTargetTrans.position.y + "," + GrabTargetTrans.position.z + "," +
            InsertionTargetTrans.position.x + "," + InsertionTargetTrans.position.y + "," + InsertionTargetTrans.position.z+","+
            IGConditionSelector.isCubeInside +"," + StartingPosition); //INCLUDES ONLY GPT OUTPUT

        markerX = 0;
    }



}