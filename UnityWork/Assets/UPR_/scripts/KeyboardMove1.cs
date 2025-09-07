using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class KeyboardMove1 : MonoBehaviour
{
    public float moveSpeed = 5.0f;
    public GameObject initialpoint;

    private void Start()
    {
        transform.position = initialpoint.transform.position;
    }

    void Update()
    {
        float moveX = 0f;
        float moveY = 0f;
        float moveZ = 0f;

        if (Input.GetKey(KeyCode.RightArrow))
        {
            moveX = -moveSpeed * Time.deltaTime;
        }
        if (Input.GetKey(KeyCode.LeftArrow))
        { 
            moveX = moveSpeed * Time.deltaTime;
        }

        if (Input.GetKey(KeyCode.DownArrow))
        {
            moveZ = moveSpeed * Time.deltaTime;
        }
        if (Input.GetKey(KeyCode.UpArrow))
        {
            moveZ = -moveSpeed * Time.deltaTime;
        }

        if (Input.GetKey(KeyCode.B)) 
        {
            moveY = moveSpeed * Time.deltaTime;
        }
        if (Input.GetKey(KeyCode.Space))
        {
            moveY = -moveSpeed * Time.deltaTime;
        }

        transform.Translate(moveX, moveY, moveZ);

    }
}
