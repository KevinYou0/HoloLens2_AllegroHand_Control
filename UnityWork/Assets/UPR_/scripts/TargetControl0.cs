using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.Utilities;

public class TargetControl0 : MonoBehaviour
{
    public Transform targetObj; // The object to follow
    public float positionThreshold = 0.01f; // Position threshold for filtering local drifting motion
    public float rotationThreshold = 0.01f; // Rotation threshold for filtering local drifting motion
    public float followSpeed = 5.0f; // Speed at which object1 follows targetObj
    public float rotationSpeed = 5.0f; // Speed at which object1 rotates to match targetObj's rotation
    public float positionFilterFactor = 0.15f; // Low-pass filter factor for position
    public float rotationFilterFactor = 0.15f; // Low-pass filter factor for rotation
    private Transform GripperControlMiddle;

    private bool solidGrip = false;

    //get finger distance
    public float fingerDistance, RealGripperDistance;

    //robot finger positions
    public GameObject leftFinger;
    public GameObject rightFinger;

    private Vector3 prevPosition;
    private Quaternion prevRotation;

    [SerializeField]
    private Handedness handedness = Handedness.Right;

    [SerializeField]
    private TrackedHandJoint handJoint = TrackedHandJoint.Palm;

    private IMixedRealityHandJointService handJointService;

    void Start()
    {
        prevPosition = new Vector3(0, 0, 0);//targetObj.position;
        prevRotation = new Quaternion(0, 0, 0, 0);//targetObj.rotation;

        handJointService = CoreServices.GetInputSystemDataProvider<IMixedRealityHandJointService>();
    }

    void Update()
    {
        if (handJointService == null)
        {
            handJointService = CoreServices.GetInputSystemDataProvider<IMixedRealityHandJointService>();
        }

        //get the transform of the hand joint
        //Transform RightHandPalmTransform = handJointService.RequestJointTransform(TrackedHandJoint.Palm, Handedness.Right);
        Transform RightIndexTip = handJointService.RequestJointTransform(TrackedHandJoint.IndexTip, Handedness.Right);
        Transform RightThumbTip = handJointService.RequestJointTransform(TrackedHandJoint.ThumbTip, Handedness.Right);

        //Transform rightHandPalmTransform = GetRightHandPalmTransform();
        Vector3 MidOfIndesandThumb = (RightIndexTip.position - RightThumbTip.position) / 2 + RightThumbTip.position;

        //Get average orientation of hand joints
        Vector3 direction = (RightIndexTip.position - RightThumbTip.position).normalized;
        Vector3 prependicular = new Vector3(-direction.z, 0, direction.x);


        fingerDistance = (RightIndexTip.position - RightThumbTip.position).magnitude;
        //print(fingerDistance);

        ////robot gripper visual (enlarged finger distance)
        //leftFinger.transform.localPosition = new Vector3(0,0,Mathf.Min((6f/6f) * fingerDistance / 2f, 5f));
        //rightFinger.transform.localPosition = new Vector3(0, 0, -Mathf.Min((6f / 6f) * fingerDistance / 2f, 5f));

        //robot gripper visual (realistic finger distance)
        leftFinger.transform.localPosition = new Vector3(0, 0, fingerDistance / 2f);
        rightFinger.transform.localPosition = new Vector3(0, 0, -fingerDistance / 2f);

        //add solid grip for insertion
        if (Input.GetKeyDown(KeyCode.G))
        {
            solidGrip = !solidGrip;
        }

        if (solidGrip == true)
        {
            RealGripperDistance = 2f;
        }
        else
        {
            RealGripperDistance = fingerDistance;
        }

        //// Calculate position and rotation differences (this method is incremental)
        //Vector3 positionDifference = MidOfIndesandThumb - prevPosition;
        //Quaternion rotationDifference = Quaternion.Inverse(prevRotation) * RightHandPalmTransform.rotation;

        //// Check if the difference is above the threshold (lock vibration will be filtered out from input)
        //if (positionDifference.magnitude > positionThreshold)
        //{
        //    prevPosition = MidOfIndesandThumb;
        //}

        //if (Quaternion.Angle(Quaternion.identity, rotationDifference) > rotationThreshold)
        //{
        //    prevRotation = RightHandPalmTransform.rotation;
        //}

        //// Apply low-pass filter to position and rotation
        //Vector3 filteredPosition = Vector3.Lerp(transform.position, prevPosition, positionFilterFactor);
        //Quaternion filteredRotation = Quaternion.Slerp(transform.rotation, prevRotation, rotationFilterFactor);

        //// Interpolate position and rotation
        //this.transform.position = Vector3.Lerp(transform.position, filteredPosition, Time.deltaTime * followSpeed);
        //this.transform.rotation = Quaternion.Slerp(transform.rotation, filteredRotation, Time.deltaTime * rotationSpeed);

        //try direct drive, without lerp 
        this.transform.position = MidOfIndesandThumb;
        //this.transform.rotation =  RightHandPalmTransform.rotation;
        //Lock rotation in x and z axis
        //this.transform.rotation = Quaternion.Euler(0f, RightHandPalmTransform.rotation.y, 0f) ;
        this.transform.rotation = Quaternion.LookRotation(prependicular,Vector3.up) ;

    }


}
