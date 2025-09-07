using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.Examples;

public class PinchGestureHandler : MonoBehaviour, IMixedRealityGestureHandler<Vector3>
{
    //public bool is_Pinching = false;
    //int count = 0;

    ////IEnumerator Reset_pinch()
    ////{
    ////    yield return new WaitForEndOfFrame();
    ////    isPinching = false;
    ////}

    //void Start()
    //{
    //    CoreServices.InputSystem?.RegisterHandler<IMixedRealityPointerHandler>(this);
    //}

    ////void Update()
    ////{
    ////    if (isPinching)
    ////    {
    ////        StartCoroutine(Reset_pinch());
    ////    }
    ////}

    //void OnDestroy()
    //{
    //    CoreServices.InputSystem?.UnregisterHandler<IMixedRealityPointerHandler>(this);
    //}

    //public void OnPointerClicked(MixedRealityPointerEventData eventData)
    //{
    //    is_Pinching = !is_Pinching;
    //}


    //public void OnPointerDown(MixedRealityPointerEventData eventData)
    //{
    //}

    //public void OnPointerUp(MixedRealityPointerEventData eventData)
    //{
    //}

    //public void OnPointerDragged(MixedRealityPointerEventData eventData) { }


    public void OnGestureStarted(InputEventData eventData)
    {
        Debug.Log($"OnGestureStarted [{Time.frameCount}]: {eventData.MixedRealityInputAction.Description}");
    }

    public void OnGestureUpdated(InputEventData eventData)
    {
        Debug.Log($"OnGestureUpdated [{Time.frameCount}]: {eventData.MixedRealityInputAction.Description}");
    }

    public void OnGestureUpdated(InputEventData<Vector3> eventData)
    {
        Debug.Log($"OnGestureUpdated [{Time.frameCount}]: {eventData.MixedRealityInputAction.Description}");
    }

    public void OnGestureCompleted(InputEventData eventData)
    {
        Debug.Log($"OnGestureCompleted [{Time.frameCount}]: {eventData.MixedRealityInputAction.Description}");
    }

    public void OnGestureCompleted(InputEventData<Vector3> eventData)
    {
        Debug.Log($"OnGestureCompleted [{Time.frameCount}]: {eventData.MixedRealityInputAction.Description}");
    }

    public void OnGestureCanceled(InputEventData eventData)
    {
        Debug.Log($"OnGestureCanceled [{Time.frameCount}]: {eventData.MixedRealityInputAction.Description}");
    }
}