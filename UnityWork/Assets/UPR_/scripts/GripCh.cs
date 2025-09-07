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
    public class GripCh : Publisher<Messages.Sensor.Joy>
    {
        public string FrameId = "Unity";
        // private ConcurrentQueue<Messages.Sensor.Joy> messageQueue = new ConcurrentQueue<Messages.Sensor.Joy>();
        private Messages.Sensor.Joy message;
        public float grip_width;

        Vector3 object_position = new Vector3(0.0f, 0.5f, 0.4f);
        Quaternion object_q = new Quaternion(0, 0, 0, 1);
        public int twist_check = 0;
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
            object_position = new Vector3(follower.transform.localPosition.x, follower.transform.localPosition.y, follower.transform.localPosition.z);
            Quaternion relativeRotation = new Quaternion(follower.transform.rotation.x, follower.transform.rotation.y, follower.transform.rotation.z, follower.transform.rotation.w);
            object_q = addRotLeft * relativeRotation;

            UpdateMessage();
        }

        private void InitializeMessage()
        {
            message = new Messages.Sensor.Joy();
            message.header.frame_id = FrameId;
            message.axes = new float[20];
            message.buttons = new int[2];
        }

        public void UpdateMessage()
        {
            message.header.Update();
            message.axes[0] = object_position.z;
            message.axes[1] = -object_position.x;
            message.axes[2] = object_position.y;

            message.axes[3] = object_q.z;
            message.axes[4] = object_q.x;
            message.axes[5] = -object_q.y;
            message.axes[6] = -object_q.w;

            message.axes[7] = grip_width;
            message.axes[8] = grip_width;

            Publish(message);
        }
    }
}