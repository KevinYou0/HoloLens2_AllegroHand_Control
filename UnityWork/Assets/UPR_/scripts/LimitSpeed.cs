using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LimitSpeed : MonoBehaviour
{
    [SerializeField]
    private float maxSpeed = 5f;

    private Vector3 previousPosition;
    private Vector3 currentVelocity;

    void Start()
    {
        previousPosition = transform.position;
    }

    void Update()
    {
        currentVelocity = (transform.position - previousPosition) / Time.deltaTime;

        if (currentVelocity.magnitude > maxSpeed)
        {
            currentVelocity = currentVelocity.normalized * maxSpeed;
            transform.position = previousPosition + currentVelocity * Time.deltaTime;
        }

        previousPosition = transform.position;
    }
}
