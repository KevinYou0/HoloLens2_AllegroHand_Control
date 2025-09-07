using System.Collections;
using System.Collections.Generic;
using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.Utilities;
using Microsoft.MixedReality.Toolkit.UI;
using UnityEngine;

public class ToggleHandMeshVisibility : MonoBehaviour
{
    private bool isHandMeshVisible = true;
    public PointerBehaviorControls pointerBehaviorControls;
    public ToggleHandVisualisation handVisualizationControls;

    // Update is called once per frame
    void Update()
    {
        //toggle visibilityof hand ray
        if (Input.GetKeyDown(KeyCode.K))
        {
            pointerBehaviorControls.ToggleHandRayEnabled();
            
        }


        //Toggle visibility of hand joints 
        if (Input.GetKeyDown(KeyCode.J))
        {
            
            handVisualizationControls.OnToggleHandJoint();
            
        }

    }

}
