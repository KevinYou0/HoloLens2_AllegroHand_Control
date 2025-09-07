//using System.Collections.Generic;
//using UnityEngine;
//using ViveSR.anipal.Eye;
//using System.Text;
//using System.IO;
//using UnityEngine.XR.Management;
//using Tobii.XR;
//using System;
//using UnityEngine.Events;
//using System.Threading;

//[System.Serializable]
//public class Recording : System.Object
//{
//    public bool showGazePoint;
//    public bool showTrail;
//}

//[System.Serializable]
//public class PlayBack : System.Object
//{
//    public bool showGazePoint;
//    public bool showTrail;
//}

//public class Row
//{
//    public int frameID;
//    public float EyePosX;      // ADDED::Eye point
//    public float EyePosY;
//    public float EyePosZ;
//    public float GazePosX;
//    public float GazePosY;
//    public float GazePosZ;
//    public float GazePointRotX;
//    public float GazePointRotY;
//    public float GazePointRotZ;
//    public float cameraPosX;
//    public float cameraPosY;
//    public float cameraPosZ;
//    public float cameraRotX;
//    public float cameraRotY;
//    public float cameraRotZ;
//    public float LeftConPosX;
//    public float LeftConPosY;
//    public float LeftConPosZ;
//    public float LeftConRotX;
//    public float LeftConRotY;
//    public float LeftConRotZ;
//    public float RightConPosX;
//    public float RightConPosY;
//    public float RightConPosZ;
//    public float RightConRotX;
//    public float RightConRotY;
//    public float RightConRotZ; // read here
//    public string LeftPupil; // 
//    public string RightPupil; // 
//    public string ObjectName;
//    public string Lumin;
//    public string UTCTime;
//    public string CurTime; //

//}

//public class DataRecCSV : MonoBehaviour
//{
//    //public UnityEvent dataRecorded = new UnityEvent();
//    UploadEyeTrackingData firebaseScript;

//    // for snake
   
//    private float forward;
//    private float turning;
//    private bool makeAturn;

//    // for sr
//    public GameObject EyeSR;
//    private VerboseData verbose_Data;

//    // for lumin
//    public GameObject Lum;
//    private double lum;

//    // recording data

//    // eye data \
//    private SingleEyeData[] eyesData;
//    private Ray rayEye;
//    private Vector3 lastGazePosition;
//    private float pd_left; //pd1
//    private float pd_right; // pd2
//    private EyeData EyeData_;


//    public string participantID;
//    public bool visualize;

//    public Recording recording;
//    public PlayBack playBack;

//    //Common to both record-time and playback visualization
//    public float visualizationDistance = 10f;
//    public Transform gazePointSpriteTransform;
//    public Sprite gazePointSprite;


//    public LineRenderer lineRenderer;

//    // csv setting 
//    private int csvLength = 45;
//    private int frameRate = 10; // originally 90

//    private int frameID = 0;
//    private float elapsedTime = 0f;
//    private string fileName = "HMD";
//    private List<string[]> csvRows = new List<string[]>();
//    private List<Row> rowList = new List<Row>();
//    private bool isLoaded = false;
//    private int frameCount = 0;
//    private float gazePointScale = 0.3f;
//    private int gazePointCloudCount = 10;
//    private int lastGazePointSpriteIndex = 0;

//    private bool spritesHidden;
//    private bool trailHidden;

//    private GameObject[] gazePointSprites;
//    private CloudPointVisualizer[] cloudPointVisualizers;

//    public GameObject LeftController;
//    public GameObject RightController;
    
//    public System.Diagnostics.Stopwatch timer;
    

//    void Start()
//    {
//        //timer = System.Diagnostics.Stopwatch.StartNew();
//        firebaseScript = gameObject.GetComponent<UploadEyeTrackingData>();

//        // time 
//        Time.fixedDeltaTime = 1f / frameRate;

//        if (visualize)
//        {
//            //UnityEngine.XR.XRSettings.enabled = false; // what is this ?
//            XRGeneralSettings.Instance.Manager.DeinitializeLoader();
//            LoadFile();
//            InitializePointCloud();
//        }
//        else
//        {
//            GetDataFromOtherComponent();
//            // test sr
//            if (!SRanipal_Eye_Framework.Instance.EnableEye)
//            {
//                enabled = false;
//                Debug.Log("please check the 'Enable Eye' box in the inspector view of SRanipal_Eye_Framework object");
//                return;
//            }
//            eyesData = new SingleEyeData[2]; // adjust? 
//            rayEye = new Ray();

//            XRGeneralSettings.Instance.Manager.InitializeLoader();
//            //PlayerSettings.virtualRealitySupported = true;
//            WriteCSVHeader();

//            //InvokeRepeating("GetDataFromOtherComponent", 0, 0.1f);

//            InvokeRepeating("RecordDataForDB", 0, 0.1f);
//        }
//    }

//    void FixedUpdate()
//    {
//        if (visualize)
//        {
//            if (isLoaded && frameID < frameCount)
//            {
//                UpdatePointCloud();
//                UpdateCameraTransform();
//                UpdateControllers();
//                //UpdateTrackers();
//                frameID++;
//                // Debug.Log("is working");
//            }
//        }
//        //else if(eyeTracker.Connected) // old
//        else
//        {
//            elapsedTime += Time.deltaTime;
//            GetDataFromOtherComponent();
//            RecordData();
//            //Debug.Log("working");
//        }

//    }

//    void OnApplicationQuit()
//    {
//        //timer.Stop();
//        //elapsedTime += timer.ElapsedMilliseconds/1000;
//        if (!visualize)
//        {
//            Debug.Log("Exiting...");
//            Debug.Log("Total recording time: " + elapsedTime + " seconds");
//            ExportCSVFile();
//        }
//    }

//    // Update is called once per frame
//    void Update()
//    {


//    }

//    void RecordDataForDB()
//    {
//        string[] sample = new string[31];

//        var data = verbose_Data; // get data from sr records
//                                 // tobii
//        var dataRay = TobiiXR.EyeTrackingData;

//        pd_left = data.left.pupil_diameter_mm;
//        pd_right = data.right.pupil_diameter_mm;

//        Vector3 eyePoint3D;                 // ADDED::Variable tracking eye point
//        Vector3 gazePoint3D;
//        string objectName = string.Empty;

//        //rayEye.direction = data.combined.eye_data.gaze_direction_normalized; // direction 
//        //rayEye.origin = data.combined.eye_data.gaze_origin_mm; // original 

//        rayEye.direction = dataRay.GazeRay.Direction; // test sr
//        rayEye.origin = dataRay.GazeRay.Origin; // test sr

//        RaycastHit hitEye;
//        if (Physics.Raycast(rayEye, out hitEye, 200))
//        {
//            //Debug.Log("Hit");
//            eyePoint3D = hitEye.point + (-0.05f) * rayEye.direction; // old                                                                              
//            objectName = hitEye.collider.name;
//            lineRenderer.SetPosition(0, rayEye.origin);
//            lineRenderer.SetPosition(1, eyePoint3D);
//        }
//        else
//           // eyePoint3D = new Vector3(0, 0, 0);
//            eyePoint3D = lastGazePosition;


//        // Raycast Gaze Direction
//        //Debug.Log(Camera.main.transform.forward.magnitude);
//        Vector3 rayDirection = Camera.main.transform.forward;
//        Vector3 rayStartPoint = Camera.main.transform.position + rayDirection;

//        Ray rayGaze = new Ray(rayStartPoint, rayDirection);          // CHANGE::Variable name ray -> rayGaze
//        RaycastHit hitGaze;
//        if (Physics.Raycast(rayGaze, out hitGaze, 200))
//        {
//            //Debug.Log("Hit");
//            gazePoint3D = hitGaze.point + (-0.05f) * rayGaze.direction; // old

//        }
//        else
//            gazePoint3D = lastGazePosition;
//        // Debug.Log(gazePoint3D);

//        Quaternion gazePointSurfaceRotation = Quaternion.identity;

//        if (gazePoint3D != Vector3.zero)
//        {

//            // new version
//            if (eyePoint3D == Vector3.zero)
//            {

//                gazePointSpriteTransform.position = lastGazePosition;
//                gazePointSurfaceRotation = Quaternion.LookRotation(hitEye.normal, Vector3.up);
//                gazePointSpriteTransform.rotation = gazePointSurfaceRotation;

//            }
//            else
//            {
//                gazePointSpriteTransform.position = eyePoint3D;
//                gazePointSurfaceRotation = Quaternion.LookRotation(hitEye.normal, Vector3.up);
//                gazePointSpriteTransform.rotation = gazePointSurfaceRotation;
//                lastGazePosition = eyePoint3D;
//            }

//        }

//        //Debug.Log("gaze:" + gazePoint3D + "eye:" + eyePoint3D);

//        if (recording.showGazePoint)
//        {
//            if (spritesHidden)
//                HideOrShowGazePoint(true);
//        }
//        else
//        {
//            if (!spritesHidden)
//                HideOrShowGazePoint(false);
//        }

//        if (recording.showTrail)
//        {
//            if (trailHidden)
//                HideOrShowTrail(true);
//        }
//        else
//        {
//            if (!trailHidden)
//                HideOrShowTrail(false);
//        }

//        // Contraints
//        // if (data.Left.GazeRayWorldValid && data.Right.GazeRayWorldValid) { 

//        sample[0] = eyePoint3D.x.ToString();
//        sample[1] = eyePoint3D.y.ToString();
//        sample[2] = eyePoint3D.z.ToString();

//        sample[3] = gazePoint3D.x.ToString();
//        sample[4] = gazePoint3D.y.ToString();
//        sample[5] = gazePoint3D.z.ToString();

//        sample[6] = gazePointSurfaceRotation.eulerAngles.x.ToString();
//        sample[7] = gazePointSurfaceRotation.eulerAngles.y.ToString();
//        sample[8] = gazePointSurfaceRotation.eulerAngles.z.ToString();

//        Vector3 cameraPosition = Camera.main.transform.position;
//        Vector3 cameraRotation = Camera.main.transform.rotation.eulerAngles;

//        sample[9] = cameraPosition.x.ToString();
//        sample[10] = cameraPosition.y.ToString();
//        sample[11] = cameraPosition.z.ToString();

//        sample[12] = cameraRotation.x.ToString();
//        sample[13] = cameraRotation.y.ToString();
//        sample[14] = cameraRotation.z.ToString();


//        /////////////////////////////// left controller/////////////////////////////////////////////////////////////
//        Vector3 LeftConPosition = LeftController.transform.position;  // left controller position and rotation
//        Vector3 LeftConRotation = LeftController.transform.rotation.eulerAngles;

//        sample[15] = LeftConPosition.x.ToString();
//        sample[16] = LeftConPosition.y.ToString();
//        sample[17] = LeftConPosition.z.ToString();

//        sample[18] = LeftConRotation.x.ToString();
//        sample[19] = LeftConRotation.y.ToString();
//        sample[20] = LeftConRotation.z.ToString();

//        ///////////////////////////////// Right controller /////////////////////////////////////////////////
//        Vector3 RightConPostion = RightController.transform.position;// right controller position and rotation
//        Vector3 RightConRotation = RightController.transform.rotation.eulerAngles;

//        sample[21] = RightConPostion.x.ToString();
//        sample[22] = RightConPostion.y.ToString();
//        sample[23] = RightConPostion.z.ToString();

//        sample[24] = RightConRotation.x.ToString();
//        sample[25] = RightConRotation.y.ToString();
//        sample[26] = RightConRotation.z.ToString();

//        sample[27] = pd_left.ToString();
//        sample[28] = pd_right.ToString();

//        ////////////////////////////////// Tracking the objects///////////////////////////////////
//        sample[29] = objectName;

//        sample[30] = lum.ToString();

//        long timeInMilli = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
//        //print(timeInMilli);

//        //dataRecorded.Invoke();
        
//        // StartCoroutine(firebaseScript.UpdateDB(sample, (long)Math.Round(timeInMilli/100.0)*100));
//        // new Thread(() => { firebaseScript.UpdateDB(sample, (long)Math.Round(timeInMilli/100.0)*100); }).Start();
//        firebaseScript.UpdateDB(sample, (long)Math.Round(timeInMilli/100.0)*100);
//    }

//    void RecordData()
//    {

//        string[] csvRow = new string[csvLength];   // 31

//        var data = verbose_Data; // get data from sr records
//                                 // tobii
//        var dataRay = TobiiXR.EyeTrackingData;

//        pd_left = data.left.pupil_diameter_mm;
//        pd_right = data.right.pupil_diameter_mm;

//        Vector3 eyePoint3D;                 // ADDED::Variable tracking eye point
//        Vector3 gazePoint3D;
//        string objectName = string.Empty;

//        //rayEye.direction = data.combined.eye_data.gaze_direction_normalized; // direction 
//        //rayEye.origin = data.combined.eye_data.gaze_origin_mm; // original 

//        rayEye.direction = dataRay.GazeRay.Direction; // test sr
//        rayEye.origin = dataRay.GazeRay.Origin; // test sr

//        RaycastHit hitEye;
//        if (Physics.Raycast(rayEye, out hitEye, 200))
//        {
//            //Debug.Log("Hit");
//            eyePoint3D = hitEye.point + (-0.05f) * rayEye.direction; // old                                                                              
//            objectName = hitEye.collider.name;
//            lineRenderer.SetPosition(0, rayEye.origin);
//            lineRenderer.SetPosition(1, eyePoint3D);
//        }
//        else
//           // eyePoint3D = new Vector3(0, 0, 0);
//            eyePoint3D = lastGazePosition;


//        // Raycast Gaze Direction
//        //Debug.Log(Camera.main.transform.forward.magnitude);
//        Vector3 rayDirection = Camera.main.transform.forward;
//        Vector3 rayStartPoint = Camera.main.transform.position + rayDirection;

//        Ray rayGaze = new Ray(rayStartPoint, rayDirection);          // CHANGE::Variable name ray -> rayGaze
//        RaycastHit hitGaze;
//        if (Physics.Raycast(rayGaze, out hitGaze, 200))
//        {
//            //Debug.Log("Hit");
//            gazePoint3D = hitGaze.point + (-0.05f) * rayGaze.direction; // old

//        }
//        else
//            gazePoint3D = lastGazePosition;
//        // Debug.Log(gazePoint3D);

//        Quaternion gazePointSurfaceRotation = Quaternion.identity;

//        if (gazePoint3D != Vector3.zero)
//        {

//            // new version
//            if (eyePoint3D == Vector3.zero)
//            {

//                gazePointSpriteTransform.position = lastGazePosition;
//                gazePointSurfaceRotation = Quaternion.LookRotation(hitEye.normal, Vector3.up);
//                gazePointSpriteTransform.rotation = gazePointSurfaceRotation;

//            }
//            else
//            {
//                gazePointSpriteTransform.position = eyePoint3D;
//                gazePointSurfaceRotation = Quaternion.LookRotation(hitEye.normal, Vector3.up);
//                gazePointSpriteTransform.rotation = gazePointSurfaceRotation;
//                lastGazePosition = eyePoint3D;
//            }

//        }

//        //Debug.Log("gaze:" + gazePoint3D + "eye:" + eyePoint3D);

//        if (recording.showGazePoint)
//        {
//            if (spritesHidden)
//                HideOrShowGazePoint(true);
//        }
//        else
//        {
//            if (!spritesHidden)
//                HideOrShowGazePoint(false);
//        }

//        if (recording.showTrail)
//        {
//            if (trailHidden)
//                HideOrShowTrail(true);
//        }
//        else
//        {
//            if (!trailHidden)
//                HideOrShowTrail(false);
//        }

//        // Contraints
//        // if (data.Left.GazeRayWorldValid && data.Right.GazeRayWorldValid) { 

//        csvRow[0] = frameID.ToString();

//        csvRow[1] = eyePoint3D.x.ToString();
//        csvRow[2] = eyePoint3D.y.ToString();
//        csvRow[3] = eyePoint3D.z.ToString();

//        csvRow[4] = gazePoint3D.x.ToString();
//        csvRow[5] = gazePoint3D.y.ToString();
//        csvRow[6] = gazePoint3D.z.ToString();

//        csvRow[7] = gazePointSurfaceRotation.eulerAngles.x.ToString();
//        csvRow[8] = gazePointSurfaceRotation.eulerAngles.y.ToString();
//        csvRow[9] = gazePointSurfaceRotation.eulerAngles.z.ToString();

//        Vector3 cameraPosition = Camera.main.transform.position;
//        Vector3 cameraRotation = Camera.main.transform.rotation.eulerAngles;

//        csvRow[10] = cameraPosition.x.ToString();
//        csvRow[11] = cameraPosition.y.ToString();
//        csvRow[12] = cameraPosition.z.ToString();

//        csvRow[13] = cameraRotation.x.ToString();
//        csvRow[14] = cameraRotation.y.ToString();
//        csvRow[15] = cameraRotation.z.ToString();


//        /////////////////////////////// left controller/////////////////////////////////////////////////////////////
//        Vector3 LeftConPosition = LeftController.transform.position;  // left controller position and rotation
//        Vector3 LeftConRotation = LeftController.transform.rotation.eulerAngles;

//        csvRow[16] = LeftConPosition.x.ToString();
//        csvRow[17] = LeftConPosition.y.ToString();
//        csvRow[18] = LeftConPosition.z.ToString();

//        csvRow[19] = LeftConRotation.x.ToString();
//        csvRow[20] = LeftConRotation.y.ToString();
//        csvRow[21] = LeftConRotation.z.ToString();

//        ///////////////////////////////// Right controller /////////////////////////////////////////////////

//        Vector3 RightConPostion = RightController.transform.position;// right controller position and rotation
//        Vector3 RightConRotation = RightController.transform.rotation.eulerAngles;

//        csvRow[22] = RightConPostion.x.ToString();
//        csvRow[23] = RightConPostion.y.ToString();
//        csvRow[24] = RightConPostion.z.ToString();

//        csvRow[25] = RightConRotation.x.ToString();
//        csvRow[26] = RightConRotation.y.ToString();
//        csvRow[27] = RightConRotation.z.ToString();

//        /////////////////////////////////// Left foot tracker /////////////////////////////////////////////////

//        //        Vector3 LeftFootTrackerPos = LeftFootTracker.transform.position;// left foot tracker position and rotation
//        //        Vector3 LeftFootTrackerRot = LeftFootTracker.transform.rotation.eulerAngles;

//        //        csvRow[28] = LeftFootTrackerPos.x.ToString();
//        //        csvRow[29] = LeftFootTrackerPos.y.ToString();
//        //        csvRow[30] = LeftFootTrackerPos.z.ToString();

//        //        csvRow[31] = LeftFootTrackerRot.x.ToString();
//        //        csvRow[32] = LeftFootTrackerRot.y.ToString();
//        //        csvRow[33] = LeftFootTrackerRot.z.ToString();
//        /////////////////////////////////// Right Foot Tracker /////////////////////////////////////////////////

//        //        Vector3 RightFootTrackerPos = RightFootTracker.transform.position;// right foot position and rotation
//        //        Vector3 RightFootTrackerRot = RightFootTracker.transform.rotation.eulerAngles;

//        //        csvRow[34] = RightFootTrackerPos.x.ToString();
//        //        csvRow[35] = RightFootTrackerPos.y.ToString();
//        //        csvRow[36] = RightFootTrackerPos.z.ToString();

//        //        csvRow[37] = RightFootTrackerRot.x.ToString();
//        //        csvRow[38] = RightFootTrackerRot.y.ToString();
//        //        csvRow[39] = RightFootTrackerRot.z.ToString();

//        /////////////////////////////////// Middle Tracker /////////////////////////////////////////////////

//        //        Vector3 MiddleTrackerPos = MidTracker.transform.position;// Middle tracker position and rotation
//        //        Vector3 MiddleTrackerRot= MidTracker.transform.rotation.eulerAngles;

//        //        csvRow[40] = MiddleTrackerPos.x.ToString();
//        //        csvRow[41] = MiddleTrackerPos.y.ToString();
//        //        csvRow[42] = MiddleTrackerPos.z.ToString();

//        //        csvRow[43] = MiddleTrackerRot.x.ToString();
//        //        csvRow[44] = MiddleTrackerRot.y.ToString();
//        //        csvRow[45] = MiddleTrackerRot.z.ToString();

//        ////////////////////////////////// pupilDiametetr///////////////////////////////////
//        // csvRow[28] = data.Left.PupilDiameter.ToString();//old
//        //  csvRow[29] = data.Right.PupilDiameter.ToString();//old

//        csvRow[28] = pd_left.ToString();
//        csvRow[29] = pd_right.ToString();

//        ////////////////////////////////// Tracking the objects///////////////////////////////////
//        csvRow[30] = objectName;

//        ////////////////////////////////// Tracking time///////////////////////////////////
//        DateTime localDate = DateTime.Now;
//        DateTime utcDate = DateTime.UtcNow;

//        csvRow[31] = lum.ToString();
//        csvRow[32] = utcDate.ToString("yyyy-MM-ddTHH:mm:ss.fff");
//        csvRow[33] = localDate.ToString("yyyy-MM-ddTHH:mm:ss.fff");
//       // Debug.Log("p1:"+pd_left.ToString()+"L1:"+lum.ToString());

//        // snake 
//        csvRow[34] = null;
//        csvRow[35] = null;
//        csvRow[36] = null;
//        csvRow[37] = null;
//        csvRow[38] = null;
//        csvRow[39] = null;
//        // control
//        csvRow[40] = forward.ToString();
//        csvRow[41] = turning.ToString();

//        csvRow[42] = makeAturn.ToString();
//        csvRows.Add(csvRow);

//        frameID++;
//        //}

//        // long timeInMilli = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
//        // print(timeInMilli);

//    }


//    void UpdateControllers()
//    {

//        Row row = rowList[frameID];

//        RightController.SetActive(true);
//        LeftController.SetActive(true);

//        ////////////////////////// left controller//////////////////////////
//        Vector3 UpdatedLeftConPos = new Vector3(row.LeftConPosX, row.LeftConPosY, row.LeftConPosZ);
//        Vector3 UpdatedLeftConRot = new Vector3(row.LeftConRotX, row.LeftConRotY, row.LeftConRotZ);

//        LeftController.transform.position = UpdatedLeftConPos;
//        LeftController.transform.rotation = Quaternion.Euler(UpdatedLeftConRot);

//        ///////////////////////////////// Right controller//////////////////////////
//        Vector3 UpdatedRightConPos = new Vector3(row.RightConPosX, row.RightConPosY, row.RightConPosZ);
//        Vector3 UpdatedRightConRot = new Vector3(row.RightConRotX, row.RightConRotY, row.RightConRotZ);

//        RightController.transform.position = UpdatedRightConPos;
//        RightController.transform.rotation = Quaternion.Euler(UpdatedRightConRot);

//    }

//    void UpdateCameraTransform()
//    {
//        Row row = rowList[frameID];
//        Vector3 cameraPosition = new Vector3(row.cameraPosX, row.cameraPosY, row.cameraPosZ);
//        Vector3 cameraRotation = new Vector3(row.cameraRotX, row.cameraRotY, row.cameraRotZ);

//        Camera.main.transform.position = cameraPosition;
//        Camera.main.transform.rotation = Quaternion.Euler(cameraRotation);
//    }

//    void UpdatePointCloud()
//    {
//        Row row = rowList[frameID];
//        Vector3 currentGazePoint = new Vector3(row.EyePosX, row.EyePosY, row.EyePosZ);   // CHANGE::EyePoint
//                                                                                         //Vector3 currentGazePoint = new Vector3(row.GazePosX, row.GazePosY, row.GazePosZ);
//        Vector3 currentTrajectoryPoint = new Vector3(row.cameraPosX, 0, row.cameraPosZ);
//        Quaternion currentGazePointRotation = Quaternion.identity;
//        Quaternion CurrentTrajectoryRotation = Quaternion.identity;
//        currentGazePointRotation.eulerAngles = new Vector3(row.GazePointRotX, row.GazePointRotY, row.GazePointRotZ);
//        CurrentTrajectoryRotation.eulerAngles = new Vector3(row.cameraRotX, row.cameraRotY, row.cameraRotZ);

//        if (currentGazePoint != Vector3.zero)
//        {
//           // gazePointSpriteTransform.position = currentGazePoint;
//            gazePointSpriteTransform.rotation = currentGazePointRotation;

//            //TrajectorySpriteTransform.position = currentTrajectoryPoint;
//            //TrajectorySpriteTransform.rotation = CurrentTrajectoryRotation;
//        }

//        if (playBack.showGazePoint)
//        {
//            if (spritesHidden)
//                HideOrShowGazePoint(true);
//        }
//        else
//        {
//            if (!spritesHidden)
//                HideOrShowGazePoint(false);
//        }

//        if (playBack.showTrail)
//        {
//            if (trailHidden)
//                HideOrShowTrail(true);
//        }
//        else
//        {
//            if (!trailHidden)
//                HideOrShowTrail(false);
//        }

//        cloudPointVisualizers[lastGazePointSpriteIndex].NewPosition(Time.time, currentGazePoint);
//        cloudPointVisualizers[lastGazePointSpriteIndex].transform.rotation = currentGazePointRotation;

//        lastGazePointSpriteIndex = (lastGazePointSpriteIndex + 1) % gazePointCloudCount;
//    }

//    void HideOrShowGazePoint(bool show)
//    {
//        if (!show)
//        {
//            if (visualize)
//            {
//                for (int i = 0; i < gazePointCloudCount; i++)
//                {
//                    cloudPointVisualizers[i]._renderer.enabled = false;
//                }
//            }
//            gazePointSpriteTransform.GetComponent<SpriteRenderer>().enabled = false;
//            //TrajectorySpriteTransform.GetComponent<SpriteRenderer>().enabled = false;
//            spritesHidden = true;
//        }
//        else
//        {
//            if (visualize)
//            {
//                for (int i = 0; i < gazePointCloudCount; i++)
//                {
//                    cloudPointVisualizers[i]._renderer.enabled = true;
//                }
//            }
//            gazePointSpriteTransform.GetComponent<SpriteRenderer>().enabled = true;
//            // TrajectorySpriteTransform.GetComponent<SpriteRenderer>().enabled = true;
//            spritesHidden = false;
//        }
//    }

//    void HideOrShowTrail(bool show)
//    {
//        if (!show)
//        {
//            gazePointSpriteTransform.GetComponent<TrailRenderer>().enabled = false;
//            //TrajectorySpriteTransform.GetComponent<SpriteRenderer>().enabled = false;
//            trailHidden = true;
//        }
//        else
//        {
//            gazePointSpriteTransform.GetComponent<TrailRenderer>().enabled = true;
//            // TrajectorySpriteTransform.GetComponent<SpriteRenderer>().enabled = true;
//            trailHidden = false;
//        }
//    }
//    public void LoadFile()
//    {
//        TextAsset file = Resources.Load("CSVOutput/" + participantID + "_" + fileName) as TextAsset;
//        Load(file);
//    }

//    public void Load(TextAsset csv) // load csv data 
//    {
//        string[][] grid = CsvParser2.Parse(csv.text);
//        frameCount = grid.Length - 1;


//        for (int i = 1; i < grid.Length; i++)
//        {
//            Row row = new Row();

//            row.frameID = int.Parse(grid[i][0]);

//            row.EyePosX = float.Parse(grid[i][1]);
//            row.EyePosY = float.Parse(grid[i][2]);
//            row.EyePosZ = float.Parse(grid[i][3]);

//            row.GazePosX = float.Parse(grid[i][4]);
//            row.GazePosY = float.Parse(grid[i][5]);
//            row.GazePosZ = float.Parse(grid[i][6]);
//            row.GazePointRotX = float.Parse(grid[i][7]);
//            row.GazePointRotY = float.Parse(grid[i][8]);
//            row.GazePointRotZ = float.Parse(grid[i][9]);

//            row.cameraPosX = float.Parse(grid[i][10]);
//            row.cameraPosY = float.Parse(grid[i][11]);
//            row.cameraPosZ = float.Parse(grid[i][12]);
//            row.cameraRotX = float.Parse(grid[i][13]);
//            row.cameraRotY = float.Parse(grid[i][14]);
//            row.cameraRotZ = float.Parse(grid[i][15]);

//            row.LeftConPosX = float.Parse(grid[i][16]);
//            row.LeftConPosY = float.Parse(grid[i][17]);
//            row.LeftConPosZ = float.Parse(grid[i][18]);
//            row.LeftConRotX = float.Parse(grid[i][19]);
//            row.LeftConRotY = float.Parse(grid[i][20]);
//            row.LeftConRotZ = float.Parse(grid[i][21]);

//            row.RightConPosX = float.Parse(grid[i][22]);
//            row.RightConPosY = float.Parse(grid[i][23]);
//            row.RightConPosZ = float.Parse(grid[i][24]);
//            row.RightConRotX = float.Parse(grid[i][25]);
//            row.RightConRotY = float.Parse(grid[i][26]);
//            row.RightConRotZ = float.Parse(grid[i][27]);


//            rowList.Add(row);
//        }

//        isLoaded = true;
//    }

//    void InitializePointCloud()
//    {
//        gazePointSprites = new GameObject[gazePointCloudCount];
//        cloudPointVisualizers = new CloudPointVisualizer[gazePointCloudCount];

//        for (int i = 0; i < gazePointCloudCount; i++)
//        {
//            GameObject newSprite = new GameObject("GazePointSprite" + i);

//            SpriteRenderer spriteRenderer = newSprite.AddComponent<SpriteRenderer>();
//            spriteRenderer.sprite = gazePointSprite;

//            CloudPointVisualizer cloudPointVisualizer = newSprite.AddComponent<CloudPointVisualizer>();
//            cloudPointVisualizer.Scale = gazePointScale;

//            gazePointSprites[i] = newSprite;
//            cloudPointVisualizers[i] = cloudPointVisualizer;
//        }
//    }

//    void WriteCSVHeader()
//    {
//        string[] header = new string[csvLength];       // 31
//        header[0] = "FrameID";
//        header[1] = "EyePosX";
//        header[2] = "EyePosY";
//        header[3] = "EyePosZ";
//        header[4] = "GazePosX";
//        header[5] = "GazePosY";
//        header[6] = "GazePosZ";
//        header[7] = "GazePointRotX";
//        header[8] = "GazePointRotY";
//        header[9] = "GazePointRotZ";
//        header[10] = "CameraPosX";
//        header[11] = "CameraPosY";
//        header[12] = "CameraPosZ";
//        header[13] = "CameraRotX";
//        header[14] = "CameraRotY";
//        header[15] = "CameraRotZ";
//        header[16] = "LeftConPosX";
//        header[17] = "LeftConPosY";
//        header[18] = "LeftConPosZ";
//        header[19] = "LeftConRotX";
//        header[20] = "LeftConRotY";
//        header[21] = "LeftConRotZ";
//        header[22] = "RightConPosX";
//        header[23] = "RightConPosY";
//        header[24] = "RightConPosZ";
//        header[25] = "RightConRotX";
//        header[26] = "RightConRotY";
//        header[27] = "RightConRotZ";
//        header[28] = "LeftPupilD";
//        header[29] = "RightPupilD";
//        header[30] = "ObjectName";
//        // add valve operation info
//        header[31] = "Lumin"; // adjust
//        header[32] = "UTC_MM"; // adjust
//        header[33] = "Local"; // adjust

//        // add snake info
//        header[34] = "Junctions"; // node
//        header[35] = "Pipes"; // spline
//        header[36] = "Rotation"; // rotation
//        header[37] = "SnakePositionX"; // trajectory 
//        header[38] = "SnakePositionY"; // trajectory 
//        header[39] = "SnakePositionZ"; // trajectory 

//        // control 
//        header[40] = "Forward";
//        header[41] = "Turning";
//        header[42] = "TurningTriggered";


//        csvRows.Add(header);
//    }

//    void ExportCSVFile()
//    {
//        Debug.Log("Exporting CSV File...");
//        string[][] output = new string[csvRows.Count][];

//        for (int i = 0; i < output.Length; i++)
//        {
//            output[i] = csvRows[i];
//        }

//        int length = output.GetLength(0);
//        string delimiter = ",";

//        StringBuilder stringBuilder = new StringBuilder();

//        for (int j = 0; j < length; j++)
//            stringBuilder.AppendLine(string.Join(delimiter, output[j]));

//        string filePath = "Assets/Resources/CSVOutput/" + participantID + "_" + fileName + ".csv";
//        Debug.Log("Wrote eye tracking data to " + filePath);
//        StreamWriter outStream = System.IO.File.CreateText(filePath);
//        outStream.Write(stringBuilder); // debug extra lines 
//        outStream.Close();
//    }

//    // Get data from outside components 
//    void GetDataFromOtherComponent()
//    {
//        verbose_Data = EyeSR.GetComponent<TestGetData>().verbose_Data;
//        lum = Lum.GetComponent<Luminosity>().lum;


//        // control
//        forward = Input.GetAxis("Vertical");
//        turning = Input.GetAxis("Horizontal");
//        makeAturn = Input.GetKeyDown("joystick button " + 0);

//    }
//}
