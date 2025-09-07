using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class move_sphere : MonoBehaviour
{
    Vector3 center_pos; // Center position
    float speed = 0.05f; // Speed of movement
    float radius = 0.05f; // Radius for circular motion in YZ plane
    private Vector3 targetPosition;
    private float angle = 0f; // Current angle of movement

    // To and fro motion variables
    float xMin = -0.3f; // Minimum X value for to-and-fro motion
    float xMax = 0.3f;  // Maximum X value for to-and-fro motion
    bool movingForward = true; // Direction flag for X-axis movement

    void Start()
    {
        center_pos = new Vector3(-0.25f, 0.581f, 0.409f);
        GenerateNewTargetPosition();
    }

    void GenerateNewTargetPosition()
    {
        float angle = Random.Range(0, 2 * Mathf.PI); // Random angle for circular motion

        // Calculate new Y and Z based on the random angle
        float y = center_pos.y + radius * Mathf.Sin(angle);
        float z = center_pos.z + radius * Mathf.Cos(angle);

        // To and fro motion along the X-axis
        float x = transform.position.x;

        // If moving forward, increase the X value; otherwise, decrease it
        if (movingForward)
        {
            x += Random.Range(0.01f, 0.05f);
            if (x >= xMax) movingForward = false; // Reverse direction when hitting max limit
        }
        else
        {
            x -= Random.Range(0.01f, 0.05f);
            if (x <= xMin) movingForward = true; // Reverse direction when hitting min limit
        }

        // Set the new target position
        targetPosition = new Vector3(x, y, z);
    }

    void Update()
    {
        /// Move the sphere within the circular area while also oscillating on the X-axis
        if (Vector3.Distance(transform.position, targetPosition) < 0.01f)
        {
            // Once the target is reached, generate a new target position
            GenerateNewTargetPosition();
        }
        else
        {
            // Move towards the target position
            transform.position = Vector3.MoveTowards(transform.position, targetPosition, speed * Time.deltaTime);
        }
    }
}
