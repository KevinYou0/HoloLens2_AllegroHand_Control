using RootMotion.FinalIK;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CCD_OnOff : MonoBehaviour
{
    public CCDIK IKToEnable1;
    public CCDIK IKToEnable2;
    // Start is called before the first frame update
    void Start()
    {
        IKToEnable1.enabled = false;
        IKToEnable2.enabled = false;
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKey("c"))
        {
            IKToEnable1.enabled = true;
            IKToEnable2.enabled = true;
        }
        if (Input.GetKey("d"))
        {
            IKToEnable1.enabled = false;
           IKToEnable2.enabled = false;
        }
    }
}
