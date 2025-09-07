using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RandomMove : MonoBehaviour
{
    public Vector3 startPoint = new Vector3(0.15f, 0.5f, 0.4f);
    public Vector3 endPoint = new Vector3(-0.15f, 0.5f, 0.4f);
    public float radiusL = 0.04f;
    public float radiusS = 0.02f;
    public float forward = 0.02f;
    public float back = 0.01f;
    public float speed = 0.01f;
    public float returnSpeed = 0.05f;
    private float rotationSpeed = 1f;
    public float angle = 5f;
    public float trial_finish = 0f;
    public int twist_check = 0;
    public float waitTime = 3f;
    public int forward_or_back= 1;

    public float arrivalTime;
    public bool isStarted = false;
    public bool isCuo = false;

    public Vector3 targetPoint;
    public Quaternion targetRotation;
    public Vector3 targetRotationEuler;
    public Transform current_start_pose;

    public float randomStartPointRangeY;
    public float randomStartPointRangeZ;
    private Vector3 oldStartPoint;

    // Start is called before the first frame update
    void Start()
    {
        oldStartPoint = startPoint;
        startPoint = RandomStartPoint();
        transform.position = startPoint;
        targetPoint = GenerateRandomPointInCylinder();
        ApplyRandomRotation();
        if (isCuo) { twist_check = GenerateRandomZeroOrOne(); } else { twist_check = 2; }
        arrivalTime = Vector3.Distance(transform.position, targetPoint) / speed;
        rotationSpeed = Quaternion.Angle(transform.rotation, targetRotation) / arrivalTime;
        current_start_pose = transform;
    }

    // Update is called once per frame
    void Update()
    {
        if (!isStarted && Input.GetKeyDown(KeyCode.Q))
        {
            isStarted = true;
        }
        else if (isStarted && Input.GetKeyDown(KeyCode.Q))
        {
            isStarted = false;
        }

        if (isCuo)
        {
            if (Input.GetKeyDown(KeyCode.T))
            {
                twist_check = 1 - twist_check;
            }
        }

        if (isStarted)
        {



            if (Vector3.Distance(transform.position, targetPoint) < 0.001f &&
            Quaternion.Angle(transform.rotation, targetRotation) < 1f)
            {
                //forward_or_back = GenerateRandomZeroOrOne();
                forward_or_back = 1;
                targetPoint = GenerateRandomPointInCylinder();
                ApplyRandomRotation();
                if (isCuo){ twist_check = GenerateRandomZeroOrOne(); } else{ twist_check = 2; }
                arrivalTime = Vector3.Distance(transform.position, targetPoint) / speed;
                rotationSpeed = Quaternion.Angle(transform.rotation, targetRotation) / arrivalTime;
                current_start_pose = transform;
            }


            if (trial_finish == 1f)
            {
                // Move back to startPoint with higher return speed
                transform.position = Vector3.MoveTowards(transform.position, startPoint, returnSpeed * Time.deltaTime);
                transform.rotation = Quaternion.Slerp(transform.rotation, Quaternion.identity, rotationSpeed * Time.deltaTime);

                // Once arrived, reset
                if (Vector3.Distance(transform.position, startPoint) < 0.001f)
                {
                    startPoint = RandomStartPoint();
                    transform.position = startPoint;
                    targetPoint = GenerateRandomPointInCylinder();
                    ApplyRandomRotation();
                    if (isCuo) { twist_check = GenerateRandomZeroOrOne(); } else { twist_check = 2; }
                    arrivalTime = Vector3.Distance(transform.position, targetPoint) / speed;
                    rotationSpeed = Quaternion.Angle(transform.rotation, targetRotation) / arrivalTime;
                    trial_finish = 0f;
                }
            }

            else
            {

                transform.position = Vector3.MoveTowards(transform.position, targetPoint, speed * Time.deltaTime);
                transform.rotation = Quaternion.Slerp(transform.rotation, targetRotation, rotationSpeed * Time.deltaTime);
            }
        }

    }
    Vector3 GenerateRandomPointInCylinder()
    {
        float angle = Random.Range(0, Mathf.PI * 2);
        float randomRadius = Random.Range(0, radiusS + (radiusL - radiusS) * ((transform.position.x - endPoint.x) / (startPoint.x - endPoint.x)));
        //float randomRadius =  radiusS + (radiusL - radiusS) * ((transform.position.x - endPoint.x)/(startPoint.x - endPoint.x));

        float y = startPoint.y + (transform.position.x - startPoint.x) / (endPoint.x - startPoint.x) * (endPoint.y - startPoint.y) + randomRadius * Mathf.Cos(angle);
        float z = startPoint.z + (transform.position.x - startPoint.x) / (endPoint.x - startPoint.x) * (endPoint.z - startPoint.z) + randomRadius * Mathf.Sin(angle);
        float x;

        if (forward_or_back == 1)
        {
            x = transform.position.x - Random.Range(0, forward);
        }
        else
        {
            x = transform.position.x + Random.Range(0, back);
            if (x > startPoint.x)
            {
                x = transform.position.x - Random.Range(0, forward);
            }
        }
        
        if (x < endPoint.x)
        {
            trial_finish = 1f;
        }
        return new Vector3(x, y, z);
    }

    Vector3 RandomStartPoint()
    {
        float x = oldStartPoint.x;
        float y = oldStartPoint.y + Random.Range(-randomStartPointRangeY, randomStartPointRangeY);
        float z = oldStartPoint.z + Random.Range(-randomStartPointRangeZ, randomStartPointRangeZ);
        return new Vector3(x, y, z);
    }
    void ApplyRandomRotation()
    {
        float randomXRotation = Random.Range(-angle, angle);
        float randomYRotation = Random.Range(-angle, angle);
        float randomZRotation = Random.Range(-angle, angle);
        targetRotationEuler = new Vector3(randomXRotation, randomYRotation, randomZRotation);
        targetRotation = Quaternion.Euler(randomXRotation, randomYRotation, randomZRotation);

    }
    int GenerateRandomZeroOrOne()
    {
        return Random.Range(0, 2);
    }

    IEnumerator WaitAndResetTrialFinish()
    {
        transform.position = startPoint;
        targetPoint = startPoint;
        yield return new WaitForSeconds(waitTime);
        trial_finish = 0f;

    }

}
