//using UnityEngine;

//using SimpleFirebaseUnity;
//using SimpleFirebaseUnity.MiniJSON;

//using System.Collections.Generic;
//using System.Collections;
//using System;
//using System.IO;

//public class UploadEyeTrackingData : MonoBehaviour
//{
//    Firebase firebase;
//    List<FirebaseQueue> firebaseQueueList;
//    int queueNum;
//    DataRecCSV eyeTrackingScript;

//    String participantID;

//    void Start()
//    {
//        firebase = Firebase.CreateNew("https://cog-dna-default-rtdb.firebaseio.com");
//        participantID = "p1";

//        firebase.OnGetSuccess += GetOKHandler;
//        firebase.OnGetFailed += GetFailHandler;
//        firebase.OnSetSuccess += SetOKHandler;
//        firebase.OnSetFailed += SetFailHandler;
//        firebase.OnUpdateSuccess += UpdateOKHandler;
//        firebase.OnUpdateFailed += UpdateFailHandler;
//        firebase.OnDeleteSuccess += DelOKHandler;
//        firebase.OnDeleteFailed += DelFailHandler;

//        firebaseQueueList = new List<FirebaseQueue>();
//        for (int i = 0; i < 5; i++)
//        {
//            firebaseQueueList.Add(new FirebaseQueue(true, 3, 1f));
//        }

//        queueNum = 0;

//        eyeTrackingScript = gameObject.GetComponent<DataRecCSV>();
//        //eyeTrackingScript.dataRecorded.AddListener(SendSample);
//        //StartCoroutine(DeleteData());
//        //StartCoroutine(Test1());
//        //StartCoroutine(Test2());
//    }

//    void GetOKHandler(Firebase sender, DataSnapshot snapshot)
//    {
//        Debug.Log("[OK] Get from key: <" + sender.FullKey + ">");
//        //Debug.Log("[OK] Raw Json: " + snapshot.RawJson);

//        Dictionary<string, object> dict = snapshot.Value<Dictionary<string, object>>();
//        List<string> keys = snapshot.Keys;

//        if (keys != null)
//            foreach (string key in keys)
//            {
//                Debug.Log(key + " = " + dict[key].ToString());
//            }
//    }
//    void GetFailHandler(Firebase sender, FirebaseError err)
//    {
//        Debug.LogError("[ERR] Get from key: <" + sender.FullKey + ">,  " + err.Message + " (" + (int)err.Status + ")");
//    }

//    void SetOKHandler(Firebase sender, DataSnapshot snapshot)
//    {
//        Debug.Log("[OK] Set from key: <" + sender.FullKey + ">" + "\n" + snapshot.RawJson);
//    }

//    void SetFailHandler(Firebase sender, FirebaseError err)
//    {
//        Debug.LogError("[ERR] Set from key: <" + sender.FullKey + ">, " + err.Message + " (" + (int)err.Status + ")");
//    }

//    void UpdateOKHandler(Firebase sender, DataSnapshot snapshot)
//    {
//        Debug.Log("[OK] Update from key: <" + sender.FullKey + ">");
//    }

//    void UpdateFailHandler(Firebase sender, FirebaseError err)
//    {
//        Debug.LogError("[ERR] Update from key: <" + sender.FullKey + ">, " + err.Message + " (" + (int)err.Status + ")");
//    }

//    void DelOKHandler(Firebase sender, DataSnapshot snapshot)
//    {
//        Debug.Log("[OK] Del from key: <" + sender.FullKey + ">");
//    }

//    void DelFailHandler(Firebase sender, FirebaseError err)
//    {
//        Debug.LogError("[ERR] Del from key: <" + sender.FullKey + ">, " + err.Message + " (" + (int)err.Status + ")");
//    }

//    void GetTimeStamp(Firebase sender, DataSnapshot snapshot)
//    {
//        long timeStamp = snapshot.Value<long>();
//        DateTime dateTime = Firebase.TimeStampToDateTime(timeStamp);

//        Debug.Log("[OK] Get on timestamp key: <" + sender.FullKey + ">");
//        Debug.Log("Date: " + timeStamp + " --> " + dateTime.ToString());
//    }

//    // DELETES ALL DATA IN "Firefighter Project" NODE
//    public IEnumerator DeleteData()
//    {
//        firebaseQueueList[queueNum].AddQueueDelete(firebase.Child("Firefighter Project"));

//        yield return null;
//    }

//    // RETRIEVES AND PRINTS OUT DATA WHEN OBSERVER DETECTS CHANGE IN DB (simplified)
//    // IEnumerator Test2()
//    // {
//    //     Firebase firebase = Firebase.CreateNew("https://icic-undergrad-research.firebaseio.com");
//    //     Firebase foo = firebase.Child("foo", true);
//    //     FirebaseQueue firebaseQueue = new FirebaseQueue(true, 3, 1f);

//    //     FirebaseObserver observer = new FirebaseObserver(firebase, 0.01f);
//    //     foo.OnGetSuccess += GetOKHandler;

//    //     observer.OnChange += (Firebase sender, DataSnapshot snapshot) =>
//    //     {
//    //         foo.GetValue(FirebaseParam.Empty.LimitToLast(3));
//    //     };
//    //     observer.Start();
//    //     Debug.Log("[OBSERVER] FirebaseObserver started!");

//    //     yield return null;
//    // }

//    /*public IEnumerator UpdateDB(String[] sampleVals, long timeInMilli)
//    {
//        var values = GetData(sampleVals);
        
//        firebaseQueue.AddQueueSet(firebase.Child("Firefighter Project", true).Child("p1", true).Child(timeInMilli.ToString(), true).Child("eyeTracking"), values);
//        //firebaseQueue.AddQueueSet(firebase.Child("Firefighter Project", true).Child(participant, true).Child(currentsample, true), GetData(values), "auth=test123"); // fix authentication issue

//        print("num seconds sent: " + (eyeTrackingScript.star*10) + " seconds\nreal time passed: " + eyeTrackingScript.watch.ElapsedMilliseconds/1000 + " seconds");

//        yield return null;
//    }*/

//    public void UpdateDB(String[] sampleVals, long timeInMilli)
//    {
//        var values = GetData(sampleVals);
//        firebaseQueueList[queueNum].AddQueueSet(firebase.Child("Firefighter Project", true).Child(participantID, true).Child("eyeTracking").Child(timeInMilli.ToString(), true), values);
//        queueNum = queueNum == 4 ? 0 : ++queueNum;
//        //print(timeInMilli);
//        // firebaseQueue.AddQueueSet(firebase.Child("Firefighter Project", true).Child("p1", true).Child(timeInMilli.ToString(), true).Child("eyeTracking"), values);
//        //firebaseQueue.AddQueueSet(firebase.Child("Firefighter Project", true).Child(participant, true).Child(currentsample, true), GetData(values), "auth=test123"); // fix authentication issue
//    }

//    // CREATES DICTIONARY OF EYE TRACKING DATA NAMES AND VALUES
//    Dictionary<string, object> GetData(String[] values)
//    {
//        Dictionary<string, object> sample = new Dictionary<string, object>();

//        // code can be refactored later to omit hard-coded values as needed
//        sample.Add("EyePosX", values[0]);
//        sample.Add("EyePosY", values[1]);
//        sample.Add("EyePosZ", values[2]);
//        sample.Add("GazePosX", values[3]);
//        sample.Add("GazePosY", values[4]);
//        sample.Add("GazePosZ", values[5]);
//        sample.Add("GazePointRotX", values[6]);
//        sample.Add("GazePointRotY", values[7]);
//        sample.Add("GazePointRotZ", values[8]);
//        sample.Add("CameraPosX", values[9]);
//        sample.Add("CameraPosY", values[10]);
//        sample.Add("CameraPosZ", values[11]);
//        sample.Add("CameraRotX", values[12]);
//        sample.Add("CameraRotY", values[13]);
//        sample.Add("CameraRotZ", values[14]);
//        sample.Add("LeftConPosX", values[15]);
//        sample.Add("LeftConPosY", values[16]);
//        sample.Add("LeftConPosZ", values[17]);
//        sample.Add("LeftConRotX", values[18]);
//        sample.Add("LeftConRotY", values[19]);
//        sample.Add("LeftConRotZ", values[20]);
//        sample.Add("RightConPosX", values[21]);
//        sample.Add("RightConPosY", values[22]);
//        sample.Add("RightConPosZ", values[23]);
//        sample.Add("RightConRotX", values[24]);
//        sample.Add("RightConRotY", values[25]);
//        sample.Add("RightConRotZ", values[26]);
//        sample.Add("LeftPupilD", values[27]);
//        sample.Add("RightPupilD", values[28]);
//        sample.Add("ObjectName", values[29]);
//        sample.Add("Lumin", values[30]);

//        return sample;
//    }
//}
