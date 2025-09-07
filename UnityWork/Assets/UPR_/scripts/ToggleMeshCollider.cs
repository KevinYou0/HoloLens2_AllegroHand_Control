using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ToggleMeshCollider : MonoBehaviour
{
    private CapsuleCollider Collider0;
    
    // Start is called before the first frame update
    void Start()
    {
        Collider0 = GetComponent<CapsuleCollider>();
    }

    public void ToggleCollider()
    {
        Collider0.enabled = !Collider0.enabled;
    }
}
