using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class test_destroy_0 : MonoBehaviour
{
    //this code is for testing the pinching functiopn with mrtk and input system
    //getting the pinching command form another code called "PinchControl0", and return the state of either pinchihng ornot
    //the pinching is not a defined gesture in MRTK, instead using "select" for the input gesture 

    
    // Reference to the PinchControl0 script on another GameObject
    public PinchControl0 pinchControl;

    void Update()
    {
        // Check if the user is pinching
        if (pinchControl.IsPinching)
        {
            Debug.Log("User is pinching");
            // Perform actions based on the pinching state
        }
        else
        {
            Debug.Log("User is not pinching");
            // Perform actions based on the non-pinching state
        }
    }
}
