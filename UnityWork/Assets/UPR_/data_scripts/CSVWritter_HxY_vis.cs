using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Text;
using System.IO;
using System;
using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.Utilities;


public class CSVWritter_HxY_vis: MonoBehaviour
{
    private StreamWriter writer;
    private Vector3 GazeHitPoint, GazeDirection, GazeOrigin, PlayerPos;
    private Quaternion PlayerRot;
    public bool ExpStarted = false;
    private bool NewData = false;
    private bool GrippedForInsetr = false;
    private int markerX = 0;

    //new variables
    [SerializeField] private GameObject Player;

    [Tooltip("00 for voice , 01 for visual")]
    public string ConditionNum;
    public string SerNum;

    //new new data recording
    public GameObject LuminSensor, PupilRecorder, DataRecorder, LumCam;

    private float LightSensorValue, PupilDiameter, PupilID, CogLoad, Trend, TrendSum, PupilAvg;
    private double LightCalculateValue;

    private IMixedRealityHandJointService handJointService;


    //switching conditions
    [Tooltip("input conditions: '00'' or '01''")]
    string Condition;

    public HelloWorld_visual ui_info;
    public CalibPos_visual moving_info;


    void Start()
    {
        handJointService = CoreServices.GetInputSystemDataProvider<IMixedRealityHandJointService>();
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.S))
        {
            Debug.Log("Experiment Started");
            NewData = true;
            InitializeCSV();
        }

        if (NewData)
        {
            WriteCSV();
        }

        if (Input.GetKeyDown(KeyCode.E))
        {
            FinishCSV();
            NewData = false;
            Debug.Log("Experiment Stopped");
        }
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
             "grasp_count," + "release_count," + "command_status," +
             "GazePointTrans X," + " GazePointTrans Y," + " GazePointTrans Z," + " GazeDirection X," + "GazeDirection Y," + "GazeDirection Z," + "GazeOrigin X," + "GazeOrigin Y," + "GazeOrigin Z,"); //INCLUDES ONLY GPT OUTPUT

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


        //set manual event input as additional marker
        if (Input.GetKey(KeyCode.X))
        {
            markerX = 1;
        }
        else
        {
            markerX = 0;
        }

        //press I to mark the startng of the experiment
        if (Input.GetKeyDown(KeyCode.S))
        {
            Debug.Log("Experiment Started");
            ExpStarted = true;
        }

        if (Input.GetKeyDown(KeyCode.E))
        {
            Debug.Log("Experiment Stopped");
            ExpStarted = false;
        }

        //mark the experiment state to gripped and insert
        if (Input.GetKeyDown(KeyCode.G))
        {
            Debug.Log("Gripped");
            GrippedForInsetr = true;
        }
        if (CoreServices.InputSystem.EyeGazeProvider.HitInfo.collider.gameObject.layer == LayerMask.NameToLayer("Spatial Awareness"))
        {
            GazeHitPoint = CoreServices.InputSystem.EyeGazeProvider.HitPosition;
        }
        GazeDirection = CoreServices.InputSystem.EyeGazeProvider.GazeDirection.normalized;
        GazeOrigin = CoreServices.InputSystem.EyeGazeProvider.GazeOrigin;


        writer.WriteLine(fileName + "," + System.DateTime.Now + "," + System.DateTime.Now.Second + "," + System.DateTime.Now.Millisecond + "," +
        Time.time + "," + Time.frameCount + "," + ExpStarted + "," + GrippedForInsetr + "," +
        PlayerPos.x + "," + PlayerPos.y + "," + PlayerPos.z + "," + PlayerRot.x + "," + PlayerRot.y + "," + PlayerRot.z + "," + PlayerRot.w + "," +
        LightSensorValue + "," + LightCalculateValue + "," + PupilDiameter + "," + PupilID + "," + CogLoad + "," + markerX + "," +
        ui_info.grasp_count + "," + ui_info.release_count + "," + moving_info.command_status + ","  +
        GazeHitPoint.x + "," + GazeHitPoint.y + "," + GazeHitPoint.z + "," + GazeDirection.x + "," + GazeDirection.y + "," + GazeDirection.z + "," + GazeOrigin.x + "," + GazeOrigin.y + "," + GazeOrigin.z); //INCLUDES ONLY GPT OUTPUT
        markerX = 0;

    }
}