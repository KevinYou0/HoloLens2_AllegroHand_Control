using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LockRotationXZ : MonoBehaviour
{
    private Quaternion initialRotation;
    

    // Update is called once per frame
    void Update()
    {
        transform.rotation = Quaternion.Euler(0, transform.eulerAngles.y, 0);
    }
}
