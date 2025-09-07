using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Luminosity : MonoBehaviour
{
    public Camera lumCam;
    Texture2D tex;
    public Camera mainCam;
    public double lum;
        
    // Start is called before the first frame update
    void Start()
    {   // this texture is a low res version of the whole scene, 
        tex = new Texture2D(lumCam.pixelWidth, lumCam.pixelHeight, TextureFormat.RGB24, false);
    }

    // Update is called once per frame
    void Update() // update //OnPostRender
    {
        //Debug.Log("average scene luminosity = " + SceneLuminosity());
        transform.position = mainCam.transform.position;
        transform.rotation = mainCam.transform.rotation;
    }

    void OnPostRender() // update //OnPostRender
    {
        lum = SceneLuminosity();
        //Debug.Log("average scene luminosity = " + SceneLuminosity());
    }

    // Compute scene luminosity as the average intensity of the image.
    // Uses standard equation for translating RGB to human perception.
    double SceneLuminosity()
    {
        tex.ReadPixels(lumCam.pixelRect, 0, 0);
        Color[] colors = tex.GetPixels();
        double total = 0;
        foreach (Color color in colors)
            total += 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;
        return total/colors.Length;
    }
}
