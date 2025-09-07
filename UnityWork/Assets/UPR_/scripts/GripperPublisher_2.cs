/*
© Siemens AG, 2017-2018
Author: Dr. Martin Bischoff (martin.bischoff@siemens.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
<http://www.apache.org/licenses/LICENSE-2.0>.
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
using UnityEngine;
using System.Collections.Generic;
using System.Collections.Concurrent; // For ConcurrentQueue

namespace RosSharp.RosBridgeClient
{
    public class GripperPublisher_2 : Publisher<Messages.Sensor.Joy>
    {
        private JoyAxisReader[] JoyAxisReaders;
        private JoyButtonReader[] JoyButtonReaders;

        public string FrameId = "Unity";
        private ConcurrentQueue<Messages.Sensor.Joy> messageQueue = new ConcurrentQueue<Messages.Sensor.Joy>();
        private Messages.Sensor.Joy message;
        private int gripper_state = 0;
        private int command = 4;
        float reset = 0f;
        float activate_gripper = 10f;

        Vector3 pos_target = new Vector3(0.511251f, -0.037871f, 0.364273f);
        Vector3 pos_small_cylinder = new Vector3(0.4715f, 0.1738f, 0.3451f);
        Vector3 pos_star = new Vector3(0.5328f, 0.1788f, 0.33005f);
        Vector3 pos_cylinder = new Vector3(0.4715f, 0.1738f, 0.3451f);


        public List<string> keyword_star = new List<string>() {"star", "top-right", "top right"};
        public List<string> keyword_cylinder = new List<string>() {"cylinder", "small cylinder"};
        public List<string> keyword_small_cylinder = new List<string>() { "Unity", "Game", "Player" };
        public List<string> keyword_square = new List<string>() { "Unity", "Game", "Player" };
        public List<string> keyword_reset = new List<string>() { "reset", "restart", "again", "over", "finished" };
        //getting the "select" gesture state form MRTK input
        //and use "Pinching" and "not Pinching" to publish the state of the controller form hand input
        //private bool Pinching, notPinching;



        protected override void Start()
        {
            base.Start();
            InitializeGameObject();
            InitializeMessage();
        }

        private void FixedUpdate()
        {
            // Process the message queue
            while (!messageQueue.IsEmpty)
            {
                if (messageQueue.TryDequeue(out Messages.Sensor.Joy message))
                {
                    Publish(message); // Publish the message
                }
            }
        }

        private void InitializeGameObject()
        {
            JoyAxisReaders = GetComponents<JoyAxisReader>();
            JoyButtonReaders = GetComponents<JoyButtonReader>();
        }

        private void InitializeMessage()
        {
            message = new Messages.Sensor.Joy();
            message.header.frame_id = FrameId;
            message.axes = new float[12];
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

        public void UpdateMessage(string target_text)
        {
            Debug.Log("publish once");
            Vector3 object_position = new Vector3(0.511251f, -0.037871f, 0.364273f);
            Vector3 target_position = new Vector3(0.511251f, -0.037871f, 0.364273f);
            string lowerCaseInputString = target_text.ToLower();

            if (CheckListForKeywords(lowerCaseInputString, keyword_star))
            {
                object_position = pos_star;
                activate_gripper = 10f;
                reset = 0f;
            }
            else if (CheckListForKeywords(lowerCaseInputString, keyword_cylinder))
            {
                object_position = pos_cylinder;
                activate_gripper = 10f;
                reset = 0f;
            }
            else if (CheckListForKeywords(lowerCaseInputString, keyword_reset))
            {
                object_position = Vector3.zero;
                target_position = Vector3.zero;
                activate_gripper = 0f;
                reset = 10f;

            }
            else
            {
                Debug.Log("No keywords found in the string.");
            }

            message.axes[0] = object_position.x;
            message.axes[1] = object_position.y;
            message.axes[2] = object_position.z;
            message.axes[3] = 1f;
            message.axes[4] = 0f;
            message.axes[5] = 0f;
            message.axes[6] = 0f;
            message.axes[7] = target_position.x;
            message.axes[8] = target_position.y;
            message.axes[9] = target_position.z;
            message.axes[10] = activate_gripper;
            message.axes[11] = reset;
            Debug.Log(object_position);
            messageQueue.Enqueue(message);
        }
    }
}
