using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.Utilities;

public class EyeExample : MonoBehaviour
{
    public GameObject dotPrefab; // Assign your dot prefab in the Inspector
    public int IsPinching = 0;
    private GameObject currentDotInstance;
    public bool IsMoving = false;
    MixedRealityPose pose_temp;

    void Update()
    {
        RaycastHit hit;
        Vector3 headPosition = Camera.main.transform.position;
        Vector3 gazeDirection = Camera.main.transform.forward;

        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.ThumbTip, Handedness.Left, out pose_temp))
        {
            IsMoving = true;
        }
        else 
        {
            IsMoving = false;
        }

        if (Physics.Raycast(headPosition, gazeDirection, out hit))
        {
            GameObject ThumbTip = GameObject.Find("ThumbTip Proxy Transform");
            GameObject IndexTip = GameObject.Find("IndexTip Proxy Transform");
            if (ThumbTip != null && IndexTip != null) 
            {
                if ((IndexTip.transform.position - ThumbTip.transform.position).magnitude < 0.05f) 
                {
                    IsPinching = 1;
                }
                if ((IndexTip.transform.position - ThumbTip.transform.position).magnitude > 0.05f)
                {
                    IsPinching = 2;
                }
            }

            // Check the name of the hit object
            string hitObjectName = hit.collider.gameObject.name;
            Debug.Log("Hit object name: " + hitObjectName);

            // Place the dot at the hit location
            if (currentDotInstance == null)
            {
                currentDotInstance = Instantiate(dotPrefab);
            }
            currentDotInstance.transform.position = hit.point;
        }
        else
        {
            if (currentDotInstance != null)
            {
                Destroy(currentDotInstance);
                currentDotInstance = null;
            }
        }
    }
}