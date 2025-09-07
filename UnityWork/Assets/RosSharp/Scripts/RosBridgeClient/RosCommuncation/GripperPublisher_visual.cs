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
    public class GripperPublisher_visual : Publisher<Messages.Sensor.Joy>
    {
        public string FrameId = "Unity";
        private ConcurrentQueue<Messages.Sensor.Joy> messageQueue = new ConcurrentQueue<Messages.Sensor.Joy>();
        private Messages.Sensor.Joy message;
        private int gripper_state = 0;
        private int command = 4;
        float reset = 0f;
        float activate_gripper = 10f;

        float graps_obj = 0f;
        float drop_obj = 0f;


        bool locker = false;
        //getting the "select" gesture state form MRTK input
        //and use "Pinching" and "not Pinching" to publish the state of the controller form hand input
        //private bool Pinching, notPinching;

        Vector3 object_position = new Vector3(-0.0f, 0.487f, 0.3f);
        Vector3 target_position = new Vector3(0.4453f, 0.2672f, 0.3229f);
        Quaternion object_q = new Quaternion(0,0,0,1);
        private Quaternion addRotLeft = Quaternion.Euler(0, 0, 180);
        public GameObject follower;

        protected override void Start()
        {
            base.Start();
            InitializeMessage();
            follower.transform.localPosition = object_position;
        }

        private void Update()
        {
            object_position = new Vector3 (follower.transform.localPosition.z, -follower.transform.localPosition.x, follower.transform.localPosition.y);
            Quaternion relativeRotation = new Quaternion(follower.transform.rotation.x, follower.transform.rotation.y, follower.transform.rotation.z, follower.transform.rotation.w);
            object_q = addRotLeft * relativeRotation;
            UpdateMessage();
        }

        private void InitializeMessage()
        {
            message = new Messages.Sensor.Joy();
            message.header.frame_id = FrameId;
            message.axes = new float[14];
            message.buttons = new int[1];
        }

        public void UpdateMessage()
        {
            message.header.Update();
            message.axes[0] = object_position.x;
            message.axes[1] = object_position.y;
            message.axes[2] = object_position.z;

            message.axes[3] = object_q.z;
            message.axes[4] = object_q.x;
            message.axes[5] = -object_q.y;
            message.axes[6] = -object_q.w;
            message.axes[7] = target_position.x;
            message.axes[8] = target_position.y;
            message.axes[9] = target_position.z;
            message.axes[10] = activate_gripper;
            message.axes[11] = reset;

            message.axes[12] = graps_obj;
            message.axes[13] = drop_obj;
            message.buttons[0] = command;
            Publish(message);
            command = 4;
        }
    }
}