using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class UpdateBrightness : MonoBehaviour
{
    public Material objectMaterial;  // Drag and drop the material of the GameObject here in the inspector
    private const float MAX_LUMINANCE = 250f;  // Adjust this value based on your observations if needed

    private void OnEnable()
    {
        // Register the method to the event
        SerialLuminanceReader.OnLuminanceReceived += HandleLuminanceReceived;
    }

    private void OnDisable()
    {
        // Unregister the method from the event
        SerialLuminanceReader.OnLuminanceReceived -= HandleLuminanceReceived;
    }

    void HandleLuminanceReceived(float luminanceValue)
    {
        // Normalize the luminance value
        float normalizedLuminance = Mathf.Clamp01(luminanceValue / MAX_LUMINANCE);

        // Apply non-linear scaling to make changes more visible
        normalizedLuminance = normalizedLuminance * normalizedLuminance;

        // Set the emission color based on the scaled luminance
        Color emissionColor = Color.yellow * normalizedLuminance;
        objectMaterial.SetColor("_EmissionColor", emissionColor);
    }
}
