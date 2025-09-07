////
//// Copyright (c) Microsoft. All rights reserved.
//// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
////

//using UnityEngine;
//using UnityEngine.UI;
//using Microsoft.CognitiveServices.Speech;
//using System;
//using System.Collections;
//using Microsoft.CognitiveServices.Speech.Audio;
//using System.IO;
//using RosSharp;
//using RosSharp.RosBridgeClient;
//using TMPro;

//using System.Collections.Generic;
//using System.Collections.Concurrent; // For ConcurrentQueue

//public class HelloWorld_visual : MonoBehaviour
//{
//    private bool micPermissionGranted = false;
//    public TextMeshProUGUI outputText;
//    public Button recoButton;
//    SpeechRecognizer recognizer;
//    SpeechConfig config;
//    AudioConfig audioInput;
//    PushAudioInputStream pushStream;

//    private object threadLocker = new object();
//    private bool recognitionStarted = false;
//    private string message;
//    int lastSample = 0;
//    AudioSource audioSource;
//    public GripperPublisher_visual rosPublisher;
//    public Vector3 MoveTo;
//    public GameObject p_handler;
//    private bool prev_pinch = false;
//    private int count = 0;

//    List<string> robot_grasp = new List<string>() { "grasp", "grip", "seize", "hold", "clutch", "catch", "snatch", "secure",
//                                                    "pick up", "clasp", "clench", "grab hold of", "take hold of", "lay hands on",
//                                                    "latch onto", "get a grip on", "envelop"};
//    List<string> robot_release = new List<string>() {"release", "let go", "unhand", "drop", "free", "loosen", "unclasp", "discharge", 
//                                                    "unleash", "liberate", "eject", "expel", "set down", "put down", "lay down", 
//                                                    "detach", "unloose", "relinquish"};
//    public List<string> keyword_reset = new List<string>() { "reset", "restart", "again", "over", "finished" };

//    public float active_gripper;
//    public float reset;
//    public int obj_lb;
//    public int target_lb;
//    public int finger_lb = 0;
//    public int grasp_count;
//    public int release_count;

//    public String detected_input;

//    private byte[] ConvertAudioClipDataToInt16ByteArray(float[] data)
//    {
//        MemoryStream dataStream = new MemoryStream();
//        int x = sizeof(Int16);
//        Int16 maxValue = Int16.MaxValue;
//        int i = 0;
//        while (i < data.Length)
//        {
//            dataStream.Write(BitConverter.GetBytes(Convert.ToInt16(data[i] * maxValue)), 0, x);
//            ++i;
//        }
//        byte[] bytes = dataStream.ToArray();
//        dataStream.Dispose();
//        return bytes;
//    }

//    private void RecognizingHandler(object sender, SpeechRecognitionEventArgs e)
//    {
//        lock (threadLocker)
//        {
//            message = e.Result.Text;
//        }
//    }

//    private void RecognizedHandler(object sender, SpeechRecognitionEventArgs e)
//    {
//        lock (threadLocker)
//        {
//            message = e.Result.Text;
//            detected_input = message;
//            Debug.Log("RecognizedHandler: " + message);
//            Debug.Log("publish once");

//            Vector3 object_position = new Vector3(0.511251f, -0.037871f, 0.364273f);
//            Vector3 target_position = new Vector3(0.511251f, -0.037871f, 0.364273f);
//            string lowerCaseInputString = message.ToLower();


//            if (p_handler.GetComponent<PinchGestureHandler>().is_Pinching != prev_pinch) 
//            {
//                count += 1;
//            }
//            if (count == 1)
//            {
//                grasp_count += 1;
//                active_gripper = 0f;
//                finger_lb = 1;
//                Debug.Log("grasp an object");
//            }
//            else if (count == 2)
//            {
//                release_count += 1;
//                active_gripper = 0f;
//                finger_lb = 0;
//                count = 0;
//                Debug.Log("release an object");
//            }

//            else if (CheckListForKeywords(lowerCaseInputString, keyword_reset))
//            {
//                obj_lb = 0;
//                target_lb = 0;
//                object_position = Vector3.zero;
//                target_position = Vector3.zero;
//                active_gripper = 0f;
//                reset = 10f;
//            }
//            else
//            {
//                Debug.Log("No keywords found in the string.");
//            }
//            prev_pinch = p_handler.GetComponent<PinchGestureHandler>().is_Pinching;
//        }
//    }

//    private void CanceledHandler(object sender, SpeechRecognitionCanceledEventArgs e)
//    {
//        lock (threadLocker)
//        {
//            message = e.ErrorDetails.ToString();
//            Debug.Log("CanceledHandler: " + message);
//        }
//    }

//    public async void ButtonClick()
//    {
//        if (recognitionStarted)
//        {
//            await recognizer.StopContinuousRecognitionAsync().ConfigureAwait(true);

//            if (Microphone.IsRecording(Microphone.devices[0]))
//            {
//                Debug.Log("Microphone.End: " + Microphone.devices[0]);
//                Microphone.End(null);
//                lastSample = 0;
//            }

//            lock (threadLocker)
//            {
//                recognitionStarted = false;
//                Debug.Log("RecognitionStarted: " + recognitionStarted.ToString());
//            }
//        }
//        else
//        {
//            if (!Microphone.IsRecording(Microphone.devices[0]))
//            {
//                Debug.Log("Microphone.Start: " + Microphone.devices[0]);
//                audioSource.clip = Microphone.Start(Microphone.devices[0], true, 200, 16000);
//                Debug.Log("audioSource.clip channels: " + audioSource.clip.channels);
//                Debug.Log("audioSource.clip frequency: " + audioSource.clip.frequency);
//            }

//            await recognizer.StartContinuousRecognitionAsync().ConfigureAwait(false);
//            lock (threadLocker)
//            {
//                recognitionStarted = true;
//                Debug.Log("RecognitionStarted: " + recognitionStarted.ToString());
//            }
//        }
//    }
//    private bool CheckListForKeywords(string inputString, List<string> keywords)
//    {
//        foreach (string keyword in keywords)
//        {
//            if (inputString.Contains(keyword))
//            {
//                return true; // A keyword was found in this list
//            }
//        }
//        return false; // No keywords found in this list
//    }
//    void Start()
//    {
//        if (outputText == null)
//        {
//            UnityEngine.Debug.LogError("outputText property is null! Assign a UI Text element to it.");
//        }
//        else
//        {
//            micPermissionGranted = true;
//            message = "";

//            config = SpeechConfig.FromSubscription("bf33e79466ad48d5869b61c54927856e", "eastus");
//            pushStream = AudioInputStream.CreatePushStream();
//            audioInput = AudioConfig.FromStreamInput(pushStream);
//            recognizer = new SpeechRecognizer(config, audioInput);
//            recognizer.Recognizing += RecognizingHandler;
//            recognizer.Recognized += RecognizedHandler;
//            recognizer.Canceled += CanceledHandler;

//            foreach (var device in Microphone.devices)
//            {
//                Debug.Log("DeviceName: " + device);                
//            }
//            audioSource = GameObject.Find("MyAudioSource").GetComponent<AudioSource>();
//        }
//    }

//    void Disable()
//    {
//        recognizer.Recognizing -= RecognizingHandler;
//        recognizer.Recognized -= RecognizedHandler;
//        recognizer.Canceled -= CanceledHandler;
//        pushStream.Close();
//        recognizer.Dispose();
//    }

//    void FixedUpdate()
//    {
//        lock (threadLocker)
//        {
//            if (outputText != null)
//            {
//                outputText.text = message;
//            }
//        }

//        if (Input.GetKeyDown(KeyCode.Space)) // Checks if the spacebar was pressed
//        {
//            ButtonClick();
//        }

//        if (Microphone.IsRecording(Microphone.devices[0]) && recognitionStarted == true)
//        {
//            Debug.Log("start converting");
//            int pos = Microphone.GetPosition(Microphone.devices[0]);
//            int diff = pos - lastSample;

//            if (diff > 0)
//            {
//                float[] samples = new float[diff * audioSource.clip.channels];
//                audioSource.clip.GetData(samples, lastSample);
//                byte[] ba = ConvertAudioClipDataToInt16ByteArray(samples);
//                if (ba.Length != 0)
//                {
//                    //Debug.Log("pushStream.Write pos:" + Microphone.GetPosition(Microphone.devices[0]).ToString() + " length: " + ba.Length.ToString());
//                    pushStream.Write(ba);
//                }
//            }
//            lastSample = pos;
//        }
//        else if (!Microphone.IsRecording(Microphone.devices[0]) && recognitionStarted == false)
//        {
//            Debug.Log("stop converting");
//        }
//    }
//}


using UnityEngine;
using UnityEngine.UI;
using RosSharp.RosBridgeClient;
using TMPro;


public class HelloWorld_visual : MonoBehaviour
{
    public GripperPublisher_visual rosPublisher;
    public Vector3 MoveTo;
    public CalibPos_visual eye_track;
    private bool prev_pinch = false;
    private int count = 0;

    public float active_gripper;
    public float reset;
    public int obj_lb;
    public int target_lb;
    public int finger_lb = 0;
    public int grasp_count;
    public int release_count;

    void FixedUpdate()
    {
        if (eye_track.IsPinching == 1)
        {
            grasp_count += 1;
            active_gripper = 0f;
            finger_lb = 1;
            Debug.Log("grasp_control");
        }
        else if (eye_track.IsPinching == 2)
        {
            release_count += 1;
            active_gripper = 0f;
            finger_lb = 0;
            Debug.Log("relase_control");
        }
    }
}