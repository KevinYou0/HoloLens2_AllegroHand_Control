using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.Utilities;


public class hand_recorder : MonoBehaviour
{

    private IMixedRealityHandJointService handJointService;

    public string HandJointRecord;

    // Start is called before the first frame update
    void Start()
    {
        handJointService = CoreServices.GetInputSystemDataProvider<IMixedRealityHandJointService>();
    }

    // Update is called once per frame
    void Update()
    {
        if (handJointService == null)
        {
            handJointService = CoreServices.GetInputSystemDataProvider<IMixedRealityHandJointService>();
        }


        string str = "";
        str = str + handJointService.RequestJointTransform(TrackedHandJoint.IndexTip, Handedness.Right) + ",";
        str = str + handJointService.RequestJointTransform(TrackedHandJoint.IndexDistalJoint, Handedness.Right) + ",";
        str = str + handJointService.RequestJointTransform(TrackedHandJoint.IndexMiddleJoint, Handedness.Right) + ",";
        str = str + handJointService.RequestJointTransform(TrackedHandJoint.IndexKnuckle, Handedness.Right) + ",";
        str = str + handJointService.RequestJointTransform(TrackedHandJoint.IndexMetacarpal, Handedness.Right) + ",";

        str = str + handJointService.RequestJointTransform(TrackedHandJoint.ThumbTip, Handedness.Right) + ",";
        str = str + handJointService.RequestJointTransform(TrackedHandJoint.ThumbDistalJoint, Handedness.Right) + ",";
        str = str + handJointService.RequestJointTransform(TrackedHandJoint.ThumbProximalJoint, Handedness.Right) + ",";
        str = str + handJointService.RequestJointTransform(TrackedHandJoint.ThumbMetacarpalJoint, Handedness.Right) + ",";

        str = str + handJointService.RequestJointTransform(TrackedHandJoint.MiddleTip, Handedness.Right) + ",";
        str = str + handJointService.RequestJointTransform(TrackedHandJoint.MiddleDistalJoint, Handedness.Right) + ",";
        str = str + handJointService.RequestJointTransform(TrackedHandJoint.MiddleMiddleJoint, Handedness.Right) + ",";
        str = str + handJointService.RequestJointTransform(TrackedHandJoint.MiddleKnuckle, Handedness.Right) + ",";
        str = str + handJointService.RequestJointTransform(TrackedHandJoint.MiddleMetacarpal, Handedness.Right) + ",";

        str = str + handJointService.RequestJointTransform(TrackedHandJoint.RingTip, Handedness.Right) + ",";
        str = str + handJointService.RequestJointTransform(TrackedHandJoint.RingDistalJoint, Handedness.Right) + ",";
        str = str + handJointService.RequestJointTransform(TrackedHandJoint.RingMiddleJoint, Handedness.Right) + ",";
        str = str + handJointService.RequestJointTransform(TrackedHandJoint.RingKnuckle, Handedness.Right) + ",";
        str = str + handJointService.RequestJointTransform(TrackedHandJoint.RingMetacarpal, Handedness.Right) + ",";

        str = str + handJointService.RequestJointTransform(TrackedHandJoint.PinkyTip, Handedness.Right) + ",";
        str = str + handJointService.RequestJointTransform(TrackedHandJoint.PinkyDistalJoint, Handedness.Right) + ",";
        str = str + handJointService.RequestJointTransform(TrackedHandJoint.PinkyMiddleJoint, Handedness.Right) + ",";
        str = str + handJointService.RequestJointTransform(TrackedHandJoint.PinkyKnuckle, Handedness.Right) + ",";
        str = str + handJointService.RequestJointTransform(TrackedHandJoint.PinkyMetacarpal, Handedness.Right) + ",";

        str = str + handJointService.RequestJointTransform(TrackedHandJoint.Palm, Handedness.Right) + ",";

        str = str + handJointService.RequestJointTransform(TrackedHandJoint.Wrist, Handedness.Right) + ",";

        HandJointRecord = str;

    }
}
