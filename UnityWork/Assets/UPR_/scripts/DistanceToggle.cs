using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DistanceToggle : MonoBehaviour
{
    // Make objectA public so it can be assigned from the Unity Editor
    public GameObject objectA;
    public GameObject objectB;

    // Make distanceThreshold public so it can be set from the Unity Editor
    public float distanceThreshold = 0.05f;

    void Update()
    {

        if (Input.GetKeyDown(KeyCode.A))
        {
            objectA.SetActive(true);
        }
        
        // Check if objectA has been assigned to avoid null reference exceptions
        if (objectA != null)
        {
            // Calculate the distance between objectA and this object (objectB)
            float distance = Vector3.Distance(objectA.transform.position, objectB.transform.position);

            // If the distance is less than the threshold, disable this object
            if (distance < distanceThreshold)
            {
                // Disable this GameObject
                objectB.SetActive(false);
            }
        }
    }
}
