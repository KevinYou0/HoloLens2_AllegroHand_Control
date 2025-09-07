using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CheckTrigger : MonoBehaviour
{
    public bool collision = false;

    // Start is called before the first frame update
    private void OnTriggerEnter(Collider other)
    {
        
        collision = true;
    }
    private void OnTriggerExit(Collider other)
    {
        collision = false;
    }
}
