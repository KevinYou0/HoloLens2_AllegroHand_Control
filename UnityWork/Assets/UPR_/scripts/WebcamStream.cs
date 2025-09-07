using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WebcamStream : MonoBehaviour
{
    void Start()
    {
        if (WebCamTexture.devices.Length > 0)
        {
            WebCamTexture webcamTexture = null;

            // Attempt to start the first working webcam
            foreach (var device in WebCamTexture.devices)
            {
                try
                {
                    webcamTexture = new WebCamTexture(device.name);
                    webcamTexture.Play(); // Attempt to start the webcam

                    // Check if the webcam has started successfully
                    if (webcamTexture.isPlaying)
                    {
                        Debug.Log("Webcam started: " + device.name);
                        break;
                    }
                    else
                    {
                        Debug.LogWarning("Could not start webcam: " + device.name);
                        webcamTexture = null; // Reset and try the next webcam
                    }
                }
                catch (System.Exception ex)
                {
                    Debug.LogError("Error starting webcam: " + device.name + " - " + ex.Message);
                    webcamTexture = null;
                }
            }

            if (webcamTexture != null && webcamTexture.isPlaying)
            {
                Renderer renderer = GetComponent<Renderer>();
                renderer.material.mainTexture = webcamTexture;
            }
            else
            {
                Debug.LogError("Failed to start any webcam.");
            }
        }
        else
        {
            Debug.LogError("No webcam detected.");
        }
    }
}
