using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Shua2 : MonoBehaviour
{
    [Tooltip("the larger object at far distance")]
    public GameObject big;

    [Tooltip("the smaller object close to user")]
    public GameObject small;

    [Tooltip("the 3rd object (independent) for rendering animation")]
    public GameObject mid;

    public Transform smallParent, bigMarker;

    //[Tooltip("the duration of shua process")]
    //public float shuaTime;

    [Tooltip("the resolution of shua process")]
    public float shuaStep;

    [Tooltip("the resolution of shua process")]
    public float shuaSpeedFactorFactor;

    private Vector3 bigTrans, bigScale, smallTrans, smallScale, midTrans, midScale, transDif, angleDif; //midTrans is where to set middle item to render, transDif is the math difference
    private Vector3 transStep, scaleStep;
    private Quaternion bigRot, smallRot, midRot, rotDif, rotStep;
    private float journeyLength, stepFraction, distCovered, startTime;
    private Renderer smallRend, bigRend, midRend;
    private bool shua1Complete = true, shua2Complete = true;
    Coroutine _shua1, _shua2;


    
    void Start()
    {
        smallRend = small.GetComponent<Renderer>();
        bigRend = big.GetComponent<Renderer>();
        midRend = mid.GetComponent<Renderer>();
        midRend.enabled = false;
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.UpArrow) && shua2Complete)
        {
            _shua1 = StartCoroutine(shua1());
        }
        if (Input.GetKeyDown(KeyCode.DownArrow) && shua1Complete)
        {
            _shua2 = StartCoroutine(shua2());
        }
    }


    public void shuaP_C() 
    {
        if (shua2Complete)
        {
            _shua1 = StartCoroutine(shua1());
        }
    }

    public void shuaC_P()
    {
        if (shua1Complete)
        {
            _shua2 = StartCoroutine(shua2());
        }
    }

    public void getPos()
    {
        bigTrans = big.transform.position;
        bigRot = big.transform.rotation;
        bigScale = big.transform.localScale;
        midTrans = mid.transform.position;
        midRot = mid.transform.rotation;
        smallTrans = small.transform.position;
        smallRot = small.transform.rotation;
        smallScale = small.transform.localScale; //lossyscale
        journeyLength = Vector3.Distance(smallTrans, bigTrans); //the distance between a abd b, always positive
        //Debug.Log("I got pos");
    }

    public void setPos1() //to be used with shua1
    {
        mid.transform.position = small.transform.position;  //set mid obj to small obj
        mid.transform.rotation = small.transform.rotation;
        mid.transform.localScale = small.transform.localScale; //lossyscale
        journeyLength = Vector3.Distance(small.transform.position, big.transform.position); //the distance between a abd b, always positive
        //Debug.Log("I set pos 1");
    }
    public void setPos2()  //to be used with shua2
    {
        mid.transform.position = big.transform.position;  //set mid obj to big obj
        mid.transform.rotation = big.transform.rotation;
        mid.transform.localScale = big.transform.localScale;
        journeyLength = Vector3.Distance(small.transform.position, big.transform.position); //the distance between a abd b, always positive
        //Debug.Log("I set pos 2");
    }

    IEnumerator shua1()    //flies from small to big
    {
        setPos1();
        smallRend.enabled = false;
        bigRend.enabled = false;
        midRend.enabled = true;
        shua1Complete = false;
        getPos();
        startTime = Time.time;
        while (Vector3.Distance(mid.transform.position, big.transform.position) > 0.1*journeyLength)
        {
            
            distCovered = (Time.time - startTime) * shuaSpeedFactorFactor;
            stepFraction = distCovered / journeyLength;
            mid.transform.position = Vector3.Lerp(smallTrans, bigTrans, Mathf.Pow(stepFraction, 5f));
            mid.transform.rotation = Quaternion.Slerp(smallRot, bigRot, Mathf.Pow(stepFraction, 5f));
            mid.transform.localScale = Vector3.Lerp(smallScale, bigScale, Mathf.Pow(stepFraction, 5f));
            yield return new WaitForFixedUpdate();
            //Debug.Log("I am still in while SPEEDFCT" + shuaSpeedFactorFactor);
        }
        //Debug.Log("I am out of while");
        setPos2();
        bigRend.enabled = true;
        midRend.enabled = false;
        shua1Complete = true;
    }

    IEnumerator shua2()    //flies from big to small
    {
        setPos2();
        smallRend.enabled = false;
        bigRend.enabled = false;
        midRend.enabled = true;
        shua2Complete = false;
        getPos();
        startTime = Time.time;
        //Debug.Log("I'm in shua 2");
        while (Vector3.Distance(mid.transform.position, small.transform.position) > 0.01f)
        {
            
            distCovered = (Time.time - startTime) * shuaSpeedFactorFactor;
            stepFraction = distCovered / journeyLength;
            mid.transform.position = Vector3.Lerp(bigTrans, smallTrans, Mathf.Pow(stepFraction, 0.2f));
            mid.transform.rotation = Quaternion.Slerp(bigRot, smallRot, Mathf.Pow(stepFraction, 0.2f));
            mid.transform.localScale = Vector3.Lerp(bigScale, smallScale, Mathf.Pow(stepFraction, 0.2f));
            yield return new WaitForFixedUpdate();
            //Debug.Log("I am still in while");
        }
        //Debug.Log("I am out of while");
        setPos1();
        smallRend.enabled = true;
        midRend.enabled = false;
        shua2Complete = true;
    }
}
