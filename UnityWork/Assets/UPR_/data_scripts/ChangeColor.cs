using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

public class ChangeColor : MonoBehaviour
{
    public GameObject object_1;
    public GameObject object_2;
    public GameObject object_3;
    public GameObject object_4;
    public GameObject object_5;
    public GameObject object_6;
    private float color;
    public float change_time = 10f;
    private float time;
    private float delta_color;


    // Start is called before the first frame update
    void Start()
    {
        delta_color = 1 / (change_time * 2);
    }

    // Update is called once per frame
    void Update()
    {
        change_color();
        time += Time.deltaTime;
    }

    void change_color()
    {
        //float delta_color = 255.0f / (change_time / 2);
        if(time<change_time)
        {
            color = 0f;
            Color colorRGB = new Color(color, color, color, 1f);
            object_1.GetComponent<Renderer>().material.SetColor("_Color", colorRGB);
            object_2.GetComponent<Renderer>().material.SetColor("_Color", colorRGB);
            object_3.GetComponent<Renderer>().material.SetColor("_Color", colorRGB);
            object_4.GetComponent<Renderer>().material.SetColor("_Color", colorRGB);
            object_5.GetComponent<Renderer>().material.SetColor("_Color", colorRGB);
            object_6.GetComponent<Renderer>().material.SetColor("_Color", colorRGB);
        }
        if (time>change_time && time<change_time*3)
        {
            color = 0f + delta_color * (time - change_time);
            Color colorRGB = new Color(color, color, color, 1f);
            object_1.GetComponent<Renderer>().material.SetColor("_Color", colorRGB);
            object_2.GetComponent<Renderer>().material.SetColor("_Color", colorRGB);
            object_3.GetComponent<Renderer>().material.SetColor("_Color", colorRGB);
            object_4.GetComponent<Renderer>().material.SetColor("_Color", colorRGB);
            object_5.GetComponent<Renderer>().material.SetColor("_Color", colorRGB);
            object_6.GetComponent<Renderer>().material.SetColor("_Color", colorRGB);
        }
        if (time > change_time*3 && time < change_time * 4)
        {
            color = 1f;
            Color colorRGB = new Color(color, color, color, 1f);
            object_1.GetComponent<Renderer>().material.SetColor("_Color", colorRGB);
            object_2.GetComponent<Renderer>().material.SetColor("_Color", colorRGB);
            object_3.GetComponent<Renderer>().material.SetColor("_Color", colorRGB);
            object_4.GetComponent<Renderer>().material.SetColor("_Color", colorRGB);
            object_5.GetComponent<Renderer>().material.SetColor("_Color", colorRGB);
            object_6.GetComponent<Renderer>().material.SetColor("_Color", colorRGB);
        }
        if (time > change_time*4 && time < change_time * 6)
        {
            color = 1f - delta_color * (time - change_time * 4);
            Color colorRGB = new Color(color, color, color, 1f);
            object_1.GetComponent<Renderer>().material.SetColor("_Color", colorRGB);
            object_2.GetComponent<Renderer>().material.SetColor("_Color", colorRGB);
            object_3.GetComponent<Renderer>().material.SetColor("_Color", colorRGB);
            object_4.GetComponent<Renderer>().material.SetColor("_Color", colorRGB);
            object_5.GetComponent<Renderer>().material.SetColor("_Color", colorRGB);
            object_6.GetComponent<Renderer>().material.SetColor("_Color", colorRGB);
        }
        if (time > change_time * 6 && time < change_time * 7)
        {
            color = 0f;
            Color colorRGB = new Color(color, color, color, 1f);
            object_1.GetComponent<Renderer>().material.SetColor("_Color", colorRGB);
            object_2.GetComponent<Renderer>().material.SetColor("_Color", colorRGB);
            object_3.GetComponent<Renderer>().material.SetColor("_Color", colorRGB);
            object_4.GetComponent<Renderer>().material.SetColor("_Color", colorRGB);
            object_5.GetComponent<Renderer>().material.SetColor("_Color", colorRGB);
            object_6.GetComponent<Renderer>().material.SetColor("_Color", colorRGB);
        }
        if(time > change_time * 7)
        {
            EditorApplication.isPlaying = false;
        }
        //print(color);
    }
}
