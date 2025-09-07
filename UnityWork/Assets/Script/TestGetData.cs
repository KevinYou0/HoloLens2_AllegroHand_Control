//using System.Collections;
//using System.Collections.Generic;
//using UnityEngine;
//using ViveSR.anipal.Eye;
//using System.Runtime.InteropServices;


//public class TestGetData : MonoBehaviour
//{
//    private static EyeData eyeData = new EyeData();
//    private bool eye_callback_registered = false;

//    public VerboseData verbose_Data; 

//    private static void EyeCallback(ref EyeData eye_data)
//    {
//        eyeData = eye_data;

//    }
//    private void Start()
//    {
//        if (!SRanipal_Eye_Framework.Instance.EnableEye)
//        {
//            enabled = false;
//            return;
//        }


//    }

//    private void Update()
//    {
//        if (SRanipal_Eye_Framework.Status != SRanipal_Eye_Framework.FrameworkStatus.WORKING &&
//            SRanipal_Eye_Framework.Status != SRanipal_Eye_Framework.FrameworkStatus.NOT_SUPPORT) return;

//        if (SRanipal_Eye_Framework.Instance.EnableEyeDataCallback == true && eye_callback_registered == false)
//        {
//            SRanipal_Eye.WrapperRegisterEyeDataCallback(Marshal.GetFunctionPointerForDelegate((SRanipal_Eye.CallbackBasic)EyeCallback));
//            eye_callback_registered = true;
//        }
//        else if (SRanipal_Eye_Framework.Instance.EnableEyeDataCallback == false && eye_callback_registered == true)
//        {
//            SRanipal_Eye.WrapperUnRegisterEyeDataCallback(Marshal.GetFunctionPointerForDelegate((SRanipal_Eye.CallbackBasic)EyeCallback));
//            eye_callback_registered = false;
//        }
//        else if (SRanipal_Eye_Framework.Instance.EnableEyeDataCallback == false)
//            SRanipal_Eye_API.GetEyeData(ref eyeData);
//        //Debug.Log(eyeData.verbose_data.left.pupil_diameter_mm);
//        verbose_Data = eyeData.verbose_data;
//}

//    private void OnDisable()
//    {
//        Release();
//    }

//    void OnApplicationQuit()
//    {
//        Release();
//    }

//    private void Release()
//    {
//        if (eye_callback_registered == true)
//        {
//            SRanipal_Eye.WrapperUnRegisterEyeDataCallback(Marshal.GetFunctionPointerForDelegate((SRanipal_Eye.CallbackBasic)EyeCallback));
//            eye_callback_registered = false;
//        }
//    }


//}
