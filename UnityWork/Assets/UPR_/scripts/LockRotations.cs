using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LockRotations : MonoBehaviour
{
    private Quaternion initialRotation;


    // Update is called once per frame
    void Update()
    {
        this.transform.rotation = Quaternion.Euler(0f, transform.rotation.y, 0f);
        //this.transform.rotation = Quaternion.Euler(0f, 0f, 0f);
    }
}