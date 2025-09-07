using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RobotStateControl : MonoBehaviour
{
    //controlling the working state and 
    public GameObject RobotSateIndicator;
    public GameObject RobotVisual;  //parent object for adjusting transparency
    public GameObject RobotLocalizedObject; //for disalbel the movement of objects

    public float transparency1 = 0.0f;
    public float transparency2 = 1f;

    public Material RobotMaterial;
    public Collider capsule0, box0;

    private void Start()
    {
        capsule0 = RobotLocalizedObject.GetComponent<CapsuleCollider>();
        box0 = RobotLocalizedObject.GetComponent<BoxCollider>();
    }

    // Update is called once per frame
    void Update()
    {
        //press L to toggle the locking state of the coliders
        if (Input.GetKeyDown(KeyCode.L))
        {
            //RobotSateIndicator.SetActive(!RobotSateIndicator.activeSelf);
            capsule0.enabled = !capsule0.enabled;
            box0.enabled = !box0.enabled;
        }

        //press I to toggle the activation status of the robot arm
        if (Input.GetKeyDown(KeyCode.I))
        {
            RobotSateIndicator.SetActive(!RobotSateIndicator.activeSelf);
            //capsule0.enabled = !capsule0.enabled;
            //box0.enabled = !box0.enabled;
        }

        if (RobotSateIndicator.activeSelf)
        {
            setMaterialTrans(RobotMaterial, transparency1);
        }
        else 
        {
            setMaterialTrans(RobotMaterial, transparency2);
        }
    }

    void setMaterialTrans(Material mat, float alpha)
    {
        Color color = mat.color;
        color.a = alpha;
        mat.color = color;
    }

}
