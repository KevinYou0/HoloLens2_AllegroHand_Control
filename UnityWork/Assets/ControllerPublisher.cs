using System.Collections;
using System.Collections.Generic;
using UnityEngine;


namespace RosSharp.RosBridgeClient
{
    public class ControllerPublisher : Publisher<Messages.Geometry.Transform>
    {

        public GameObject targetJoint;

        //an object that is used to (active state) to lock/unlock the movement of robot 
        public GameObject stateIndicator;

        private Messages.Geometry.Transform message;

        private string grapHandtype;

        private bool start = false;

        private Quaternion ModRotLeft, addRotLeft = Quaternion.Euler(0,0,180);

        protected override void Start()
        {
            base.Start();
            InitializeMessage();
        }


        //private void Update()
        //{
        //    Debug.Log("ROATION   X="+ targetJoint.transform.rotation.x + "Y=" + targetJoint.transform.rotation.y + "Z="+ targetJoint.transform.rotation.z +"W="+ targetJoint.transform.rotation.w);
        //}

        private void FixedUpdate()
        {

            if (stateIndicator.activeSelf)
            {
                if (Time.frameCount % 1 == 0)
                {
                    ModRotLeft = addRotLeft * targetJoint.transform.rotation;

                    UpdateMessage();
                }
            }

        }

        private void InitializeMessage()
        {
            message = new Messages.Geometry.Transform();
            message.translation = new Messages.Geometry.Vector3();
            message.rotation = new Messages.Geometry.Quaternion();
        }
        private void UpdateMessage()
        {
            //currentHand = interactableGrap.currentHand;


            message.translation = GetGeometryVector3(targetJoint.transform.localPosition);
            //message.rotation = GetGeometryVector4(targetJoint.transform.rotation);
            message.rotation = GetGeometryVector4(ModRotLeft);

            Publish(message);

        }

        private static Messages.Geometry.Vector3 GetGeometryVector3(Vector3 vector3)
        {
            Messages.Geometry.Vector3 geometryVector3 = new Messages.Geometry.Vector3();
            geometryVector3.x = vector3.x;
            geometryVector3.y = vector3.y;
            geometryVector3.z = vector3.z;
            return geometryVector3;
        }

        private static Messages.Geometry.Quaternion GetGeometryVector4(Quaternion vector4)
        {
            Messages.Geometry.Quaternion geometryVector4 = new Messages.Geometry.Quaternion();
            geometryVector4.x = vector4.x;
            geometryVector4.y = vector4.y;
            geometryVector4.z = vector4.z;
            geometryVector4.w = vector4.w;
            return geometryVector4;
        }

    }
}