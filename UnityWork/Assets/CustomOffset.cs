using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CustomOffset : MonoBehaviour
{
    public Transform target;
    public Transform wholescale;
    public Vector3 PositionOffset;
    //public Vector3 RotationOffset = new Vector3(0, 0, 0);
    private bool scaled = false;
    private Vector3 ScaledOffset;
    private float initialScale;

    // Start is called before the first frame update
    void Start()
    {
        transform.position = target.position + PositionOffset;
        //transform.Rotate(RotationOffset);
        initialScale = wholescale.localScale[0];
        ScaledOffset = PositionOffset;
    }

    // Update is called once per frame
    void Update()
    {
        if (!scaled && initialScale != wholescale.localScale[0]){
            ScaledOffset = wholescale.localScale[0]/initialScale * PositionOffset;
            scaled = true;
            Debug.Log("scaled to " + ScaledOffset);
        }
        Vector3 newoffset = target.transform.right * ScaledOffset[0]+ target.transform.up * ScaledOffset[1] + target.transform.forward * ScaledOffset[2];
        transform.position = newoffset + target.position;
        //transform.Rotate(RotationOffset);
    }


}
