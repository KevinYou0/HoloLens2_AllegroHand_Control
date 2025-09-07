using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.Input;

public class PinchControl0 : MonoBehaviour, IMixedRealityPointerHandler
{
    private bool isPinching = false;

    public bool IsPinching
    {
        get { return isPinching; }
    }

    void Start()
    {
        CoreServices.InputSystem?.RegisterHandler<IMixedRealityPointerHandler>(this);
    }

    void OnDestroy()
    {
        CoreServices.InputSystem?.UnregisterHandler<IMixedRealityPointerHandler>(this);
    }

    public void OnPointerClicked(MixedRealityPointerEventData eventData) { }

    public void OnPointerDown(MixedRealityPointerEventData eventData)
    {
        if (eventData.MixedRealityInputAction.Description == "Select")
        {
            isPinching = true;
        }
    }

    public void OnPointerUp(MixedRealityPointerEventData eventData)
    {
        if (eventData.MixedRealityInputAction.Description == "Select")
        {
            isPinching = false;
        }
    }

    public void OnPointerDragged(MixedRealityPointerEventData eventData) { }
}