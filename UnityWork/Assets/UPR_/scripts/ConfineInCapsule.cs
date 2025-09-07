using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ConfineInCapsule : MonoBehaviour
{
    [SerializeField] private Transform objectToConfine;
    [SerializeField] private Transform capsule;

    private void Update()
    {
        Vector3 confinedPosition = GetConfinedPosition(objectToConfine.position);
        objectToConfine.position = confinedPosition;
    }

    private Vector3 GetConfinedPosition(Vector3 position)
    {
        // Calculate the position in the Capsule's local space
        Vector3 localPosition = capsule.InverseTransformPoint(position);

        // Calculate the center of the bottom of the capsule
        Vector3 bottomCenter = new Vector3(0, 0.5f, 0);

        // Calculate the distance between the position and the bottom center of the capsule
        Vector3 offset = localPosition - bottomCenter;
        float distance = offset.magnitude;

        // Check if the object is within the capsule radius
        if (distance > 0.5f)
        {
            // Clamp the object's position within the capsule radius
            localPosition = bottomCenter + (offset / distance) * 0.5f;
        }

        // Check if the object is within the half-capsule height
        if (localPosition.z < 0)
        {
            localPosition.z = 0;
        }

        // Convert the confined position back to world space
        Vector3 worldPosition = capsule.TransformPoint(localPosition);

        return worldPosition;
    }
}
