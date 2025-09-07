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

namespace RosSharp.RosBridgeClient
{
    public class GripperPublisher : Publisher<Messages.Sensor.Joy>
    {
        private JoyAxisReader[] JoyAxisReaders;
        private JoyButtonReader[] JoyButtonReaders;

        public string FrameId = "Unity";

        private Messages.Sensor.Joy message;
        private int gripper_state = 0;
        private int command = 4;
        float dis = 0f;
        float homing = 0f;
        //getting the "select" gesture state form MRTK input
        //and use "Pinching" and "not Pinching" to publish the state of the controller form hand input
        public PinchControl0 pinchControl;
        //private bool Pinching, notPinching;

        public GameObject targetJoint;
        public GameObject stateIndicator;
        public GameObject Capsule_big;
        public TargetControl0 dis_obj;
        private Quaternion ModRotLeft, addRotLeft = Quaternion.Euler(0, 0, 180);
        private Quaternion CapsuleRotation;

        protected override void Start()
        {
            base.Start();
            InitializeGameObject();
            InitializeMessage();
        }

        Quaternion GetRelativeRotation(Transform reference, Transform target)
        {
            return Quaternion.Inverse(reference.rotation) * target.rotation;
        }

        private void FixedUpdate()
        {
            CapsuleRotation = Capsule_big.transform.rotation;


            if (stateIndicator.activeSelf)
            {
                if (Time.frameCount % 1 == 0)
                {
                    //ModRotLeft = addRotLeft * targetJoint.transform.rotation;
                    Quaternion relativeRotation = GetRelativeRotation(Capsule_big.transform, targetJoint.transform);

                    ModRotLeft = addRotLeft * relativeRotation;

                    dis = dis_obj.RealGripperDistance;
                    UpdateMessage();
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
            //message.axes = new float[JoyAxisReaders.Length];
            message.axes = new float[9];
            ////message.buttons = new int[JoyButtonReaders.Length];
            //message.buttons = new int[9];
        }

        private void UpdateMessage()
        {

            ////if (Input.GetKey("1") & gripper_state == 0)
            ////{
            ////    command = 1;
            ////    gripper_state = 1;
            ////}
            ////if (Input.GetKey("2") & gripper_state == 1)
            ////{
            ////    command = 2;
            ////    gripper_state = 0;
            ////}
            ////if (Input.GetKey("3"))
            ////{
            ////    command = 3;
            ////}

            //if (pinchControl.IsPinching && gripper_state == 0)
            //{
            //    command = 1;
            //    gripper_state = 1;
            //}
            //if (!pinchControl.IsPinching && gripper_state == 1)
            //{
            //    command = 2;
            //    gripper_state = 0;
            //}
            //if (Input.GetKey("3"))
            //{
            //    command = 3;
            //}

            //message.header.Update();

            //for (int i = 0; i < JoyAxisReaders.Length; i++)
            //    message.axes[i] = JoyAxisReaders[i].Read();

            //for (int i = 0; i < JoyButtonReaders.Length; i++)
            //    message.buttons[i] = (JoyButtonReaders[i].Read() ? 1 : 0);
            message.axes[0] = targetJoint.transform.localPosition.z;
            message.axes[1] = -targetJoint.transform.localPosition.x;
            message.axes[2] = targetJoint.transform.localPosition.y;
            message.axes[3] = ModRotLeft.z;
            message.axes[4] = ModRotLeft.x;
            message.axes[5] = -ModRotLeft.y;
            message.axes[6] = -ModRotLeft.w;
            message.axes[7] = homing;
            message.axes[8] = dis;
            //message.buttons[0] = command;
            Publish(message);
            ////Debug.Log(message);
            ////Debug.Log(command);
            //command = 4;
        }
    }
}
