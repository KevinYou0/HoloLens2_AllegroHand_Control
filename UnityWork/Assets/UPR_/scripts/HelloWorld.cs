//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

using UnityEngine;
using UnityEngine.UI;
using Microsoft.CognitiveServices.Speech;
using System;
using System.Collections;
using Microsoft.CognitiveServices.Speech.Audio;
using System.IO;
using RosSharp;
using RosSharp.RosBridgeClient;
using TMPro;

using System.Collections.Generic;
using System.Collections.Concurrent; // For ConcurrentQueue

public class HelloWorld : MonoBehaviour
{
    private bool micPermissionGranted = false;
    public TextMeshProUGUI outputText;
    public Button recoButton;
    SpeechRecognizer recognizer;
    SpeechConfig config;
    AudioConfig audioInput;
    PushAudioInputStream pushStream;

    private object threadLocker = new object();
    private bool recognitionStarted = false;
    private string message;
    int lastSample = 0;
    AudioSource audioSource;
    public GripperPublisher_demo rosPublisher;
    public CalibPos moving_control;
    public Vector3 MoveTo;

    List<string> keyword_star = new List<string>() { "star"};
    List<string> keyword_cylinder = new List<string>() {"big", "cylinder", "left"};
    List<string> keyword_small_cylinder = new List<string>() { "small", "right", "square", "cube", "brick"};

    List<string> keyword_target_close = new List<string>() { "first", "front", "close"};
    List<string> keyword_target_mid = new List<string>() { "second", "center", "middle"};
    List<string> keyword_target_far = new List<string>() { "third", "back", "far"};

    List<string> robot_grasp = new List<string>() { "grasp", "grip", "seize", "hold", "clutch", "catch", "snatch", "secure",
                                                    "pick up", "clasp", "clench", "grab hold of", "take hold of", "lay hands on",
                                                    "latch onto", "get a grip on", "envelop"};
    List<string> robot_release = new List<string>() {"release", "let go", "unhand", "drop", "free", "loosen", "unclasp", "discharge", 
                                                    "unleash", "liberate", "eject", "expel", "set down", "put down", "lay down", 
                                                    "detach", "unloose", "relinquish"};

    public List<string> keyword_reset = new List<string>() { "reset", "restart", "again", "over", "finished" };

    public float active_gripper;
    public float reset;
    public int obj_lb;
    public int target_lb;
    public int finger_lb = 0;
    public int move_count;
    public int target_count;
    public int grasp_count;
    public int release_count;

    public String detected_input;

    private byte[] ConvertAudioClipDataToInt16ByteArray(float[] data)
    {
        MemoryStream dataStream = new MemoryStream();
        int x = sizeof(Int16);
        Int16 maxValue = Int16.MaxValue;
        int i = 0;
        while (i < data.Length)
        {
            dataStream.Write(BitConverter.GetBytes(Convert.ToInt16(data[i] * maxValue)), 0, x);
            ++i;
        }
        byte[] bytes = dataStream.ToArray();
        dataStream.Dispose();
        return bytes;
    }

    private void RecognizingHandler(object sender, SpeechRecognitionEventArgs e)
    {
        moving_control.command_status = 1;
        lock (threadLocker)
        {
            message = e.Result.Text;
        }
    }

    private void RecognizedHandler(object sender, SpeechRecognitionEventArgs e)
    {
        lock (threadLocker)
        {
            message = e.Result.Text;
            detected_input = message;
            Debug.Log("RecognizedHandler: " + message);
            Debug.Log("publish once");

            Vector3 object_position = new Vector3(0.511251f, -0.037871f, 0.364273f);
            Vector3 target_position = new Vector3(0.511251f, -0.037871f, 0.364273f);
            string lowerCaseInputString = message.ToLower();

            if (CheckListForKeywords(lowerCaseInputString, keyword_small_cylinder) || Input.GetKeyDown(KeyCode.L))
            {
                move_count += 1;
                active_gripper = 10f;
                obj_lb = 1;
                print("press once");
            }
            else if (CheckListForKeywords(lowerCaseInputString, keyword_star) || Input.GetKeyDown(KeyCode.M))
            {
                move_count += 1;
                active_gripper = 10f;
                obj_lb = 2;
            }
            else if (CheckListForKeywords(lowerCaseInputString, keyword_cylinder) || Input.GetKeyDown(KeyCode.R))
            {
                move_count += 1;
                active_gripper = 10f;
                obj_lb = 3;
            }
            else if (CheckListForKeywords(lowerCaseInputString, keyword_target_close) || Input.GetKeyDown(KeyCode.Q))
            {
                active_gripper = 10f;
                target_lb = 4;
            }
            else if (CheckListForKeywords(lowerCaseInputString, keyword_target_mid) || Input.GetKeyDown(KeyCode.W))
            {
                target_count += 1;
                active_gripper = 10f;
                target_lb = 5;
            }
            else if (CheckListForKeywords(lowerCaseInputString, keyword_target_far) || Input.GetKeyDown(KeyCode.T))
            {
                target_count += 1;
                active_gripper = 10f;
                target_lb = 6;
            }
            else if (CheckListForKeywords(lowerCaseInputString, robot_grasp) || Input.GetKeyDown(KeyCode.G))
            {
                grasp_count += 1;
                active_gripper = 0f;
                finger_lb = 1;
            }
            else if (CheckListForKeywords(lowerCaseInputString, robot_release) || Input.GetKeyDown(KeyCode.D))
            {
                release_count += 1;
                active_gripper = 0f;
                finger_lb = 0;
            }

            else if (CheckListForKeywords(lowerCaseInputString, keyword_reset))
            {
                obj_lb = 0;
                target_lb = 0;
                object_position = Vector3.zero;
                target_position = Vector3.zero;
                active_gripper = 0f;
                reset = 10f;

            }
            else
            {
                Debug.Log("No keywords found in the string.");
            }

        }
    }

    private void CanceledHandler(object sender, SpeechRecognitionCanceledEventArgs e)
    {
        lock (threadLocker)
        {
            message = e.ErrorDetails.ToString();
            Debug.Log("CanceledHandler: " + message);
        }
    }

    public async void ButtonClick()
    {
        if (recognitionStarted)
        {
            await recognizer.StopContinuousRecognitionAsync().ConfigureAwait(true);
           

            if (Microphone.IsRecording(Microphone.devices[0]))
            {
                Debug.Log("Microphone.End: " + Microphone.devices[0]);
                Microphone.End(null);
                lastSample = 0;
            }

            lock (threadLocker)
            {
                recognitionStarted = false;
                Debug.Log("RecognitionStarted: " + recognitionStarted.ToString());
            }
        }
        else
        {
            if (!Microphone.IsRecording(Microphone.devices[0]))
            {
                Debug.Log("Microphone.Start: " + Microphone.devices[0]);
                audioSource.clip = Microphone.Start(Microphone.devices[0], true, 200, 16000);
                Debug.Log("audioSource.clip channels: " + audioSource.clip.channels);
                Debug.Log("audioSource.clip frequency: " + audioSource.clip.frequency);
            }

            await recognizer.StartContinuousRecognitionAsync().ConfigureAwait(false);
            lock (threadLocker)
            {
                recognitionStarted = true;
                Debug.Log("RecognitionStarted: " + recognitionStarted.ToString());
            }
        }
    }
    private bool CheckListForKeywords(string inputString, List<string> keywords)
    {
        foreach (string keyword in keywords)
        {
            if (inputString.Contains(keyword))
            {
                return true; // A keyword was found in this list
            }
        }
        return false; // No keywords found in this list
    }
    void Start()
    {
        if (outputText == null)
        {
            UnityEngine.Debug.LogError("outputText property is null! Assign a UI Text element to it.");
        }
        else
        {
            micPermissionGranted = true;
            message = "";

            config = SpeechConfig.FromSubscription("bf33e79466ad48d5869b61c54927856e", "eastus");
            pushStream = AudioInputStream.CreatePushStream();
            audioInput = AudioConfig.FromStreamInput(pushStream);
            recognizer = new SpeechRecognizer(config, audioInput);
            recognizer.Recognizing += RecognizingHandler;
            recognizer.Recognized += RecognizedHandler;
            recognizer.Canceled += CanceledHandler;

            //recoButton.onClick.AddListener(ButtonClick);
            foreach (var device in Microphone.devices)
            {
                Debug.Log("DeviceName: " + device);                
            }
            audioSource = GameObject.Find("MyAudioSource").GetComponent<AudioSource>();
        }
    }

    void Disable()
    {
        recognizer.Recognizing -= RecognizingHandler;
        recognizer.Recognized -= RecognizedHandler;
        recognizer.Canceled -= CanceledHandler;
        pushStream.Close();
        recognizer.Dispose();
    }

    void FixedUpdate()
    {

        lock (threadLocker)
        {
            if (outputText != null)
            {
                outputText.text = message;
            }
        }

        if (Input.GetKeyDown(KeyCode.Space)) // Checks if the spacebar was pressed
        {
            ButtonClick();
        }

        if (Input.GetKeyDown(KeyCode.L))
        {
            move_count += 1;
            active_gripper = 10f;
            obj_lb = 1;
            print("press once");
        }
        else if (Input.GetKeyDown(KeyCode.M))
        {
            move_count += 1;
            active_gripper = 10f;
            obj_lb = 2;
        }
        else if (Input.GetKeyDown(KeyCode.R))
        {
            move_count += 1;
            active_gripper = 10f;
            obj_lb = 3;
        }
        else if (Input.GetKeyDown(KeyCode.Q))
        {
            active_gripper = 10f;
            target_lb = 4;
        }
        else if (Input.GetKeyDown(KeyCode.W))
        {
            target_count += 1;
            active_gripper = 10f;
            target_lb = 5;
        }
        else if (Input.GetKeyDown(KeyCode.T))
        {
            target_count += 1;
            active_gripper = 10f;
            target_lb = 6;
        }
        else if (Input.GetKeyDown(KeyCode.G))
        {
            grasp_count += 1;
            active_gripper = 0f;
            finger_lb = 1;
        }
        else if (Input.GetKeyDown(KeyCode.D))
        {
            release_count += 1;
            active_gripper = 0f;
            finger_lb = 0;
        }


        if (Microphone.IsRecording(Microphone.devices[0]) && recognitionStarted == true)
        {
            Debug.Log("start converting");
            int pos = Microphone.GetPosition(Microphone.devices[0]);
            int diff = pos - lastSample;

            if (diff > 0)
            {
                float[] samples = new float[diff * audioSource.clip.channels];
                audioSource.clip.GetData(samples, lastSample);
                byte[] ba = ConvertAudioClipDataToInt16ByteArray(samples);
                if (ba.Length != 0)
                {
                    pushStream.Write(ba);
                }
            }
            lastSample = pos;
        }
        else if (!Microphone.IsRecording(Microphone.devices[0]) && recognitionStarted == false)
        {
            Debug.Log("stop converting");
        }
    }
}
