using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.SpatialAwareness;


public class SpatialMeshManager : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        DisableCollision();
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    void DisableCollision()
    {
        var spatialAwarenessSystem = CoreServices.SpatialAwarenessSystem;

        if (spatialAwarenessSystem == null)
        {
            Debug.LogError("Spatial Awareness System is not found.");
            return;
        }

        var observers = spatialAwarenessSystem.GetObservers<IMixedRealitySpatialAwarenessMeshObserver>();

        foreach (var observer in observers)
        {
            observer.DisplayOption = SpatialAwarenessMeshDisplayOptions.None;
        }

    }


}
