using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SphereVisibilityController : MonoBehaviour
{
    // Public field to assign the object that can trigger the mesh renderer to disable
    public GameObject InvasionObject;

    private MeshRenderer meshRenderer;

    private void Start()
    {
        // Get the MeshRenderer component of the sphere
        meshRenderer = GetComponent<MeshRenderer>();

        // Ensure the attached collider is marked as a trigger
        Collider collider = GetComponent<Collider>();
    }

    private void OnTriggerEnter(Collider other)
    {
        // Check if the entering object is the InvasionObject
        if (other.gameObject == InvasionObject)
        {
            // Disable the MeshRenderer
            meshRenderer.enabled = false;
        }
    }

    private void OnTriggerExit(Collider other)
    {
        // Check if the exiting object is the InvasionObject
        if (other.gameObject == InvasionObject)
        {
            // Enable the MeshRenderer
            meshRenderer.enabled = true;
        }
    }
}
