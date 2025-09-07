using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using RosSharp.RosBridgeClient;
public class CalibPos : MonoBehaviour
{
    public GameObject ee_hand;
    public GameObject tracker;
    public JoySubscriber_vs reader;
    public HelloWorld obj_ind;
    public bool reach_obj = false;
    public bool reach_tg = false;
    public int command_status = 0; //once receive, set to 1, during operation, set to 2, 

    Vector3 startPoint;
    Vector3 controlPoint1;
    Vector3 controlPoint2;
    Vector3 endPoint;
    public Vector3 ee_hand_pos;
    
    // this is in the robot arm system
    Vector3 pos_cylinder = new Vector3(0.52094144f, -0.11900946f, 0.344116638f);
    Vector3 pos_star = new Vector3(0.52141990f, 0.00945116f, 0.34992781f);
    Vector3 pos_small_cylinder = new Vector3(0.52100867f, 0.12295886f, 0.34849644f);

    Vector3 target_position_close = new Vector3(0.44065475f, 0.28345066f, 0.37329465f);
    Vector3 target_position_mid = new Vector3(0.56062072f, 0.28442630f, 0.35793865f);
    Vector3 target_position_far = new Vector3(0.67626953f, 0.27326885f, 0.37563881f);
    private bool isMoving = false;
    private float speed = 0.1f;
    private float startTime;
    private float journeyLength;
    private int phase = 0;

    void Start()
    {
        GameObject star = GameObject.Find("star.Shrink");
        star.transform.localPosition = new Vector3 (-pos_star.y, pos_star.z, pos_star.x);
        GameObject Cyl_big = GameObject.Find("Cyl_big.Shrink");
        Cyl_big.transform.localPosition = new Vector3(-pos_cylinder.y, pos_cylinder.z, pos_cylinder.x);
        GameObject Cyl_sml = GameObject.Find("Cyl_small.Shrink");
        Cyl_sml.transform.localPosition = new Vector3(-pos_small_cylinder.y, pos_small_cylinder.z, pos_small_cylinder.x);
    }

        // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.C))
        {
            print("C");
            tracker.transform.localPosition = new Vector3(reader.myArray[0], reader.myArray[1], reader.myArray[2]);
        }

        /// move to the objects
        if (obj_ind.obj_lb == 1 && !isMoving) 
        {
            command_status = 2;
            print("L");
            startPoint = tracker.transform.localPosition;
            controlPoint1 = new Vector3(tracker.transform.localPosition.x, tracker.transform.localPosition.y + 0.16f, tracker.transform.localPosition.z);
            controlPoint2 = new Vector3(-pos_small_cylinder.y, pos_small_cylinder.z + 0.16f, pos_small_cylinder.x);
            endPoint = new Vector3(-pos_small_cylinder.y, pos_small_cylinder.z, pos_small_cylinder.x);
            // Start the movement when button is clicked
            if (!isMoving)
            {
                startTime = Time.time;
                isMoving = true;
                phase = 1;
            }
            reach_obj = false;
        }
        if (obj_ind.obj_lb == 2 && !isMoving)
        {
            command_status = 2;
            print("M");
            startPoint = tracker.transform.localPosition;
            controlPoint1 = new Vector3(tracker.transform.localPosition.x, tracker.transform.localPosition.y + 0.16f, tracker.transform.localPosition.z);
            controlPoint2 = new Vector3(-pos_star.y, pos_star.z + 0.16f, pos_star.x);
            endPoint = new Vector3(-pos_star.y, pos_star.z, pos_star.x);
            // Start the movement when button is clicked
            if (!isMoving)
            {
                startTime = Time.time;
                isMoving = true;
                phase = 1;
            }
            reach_obj = false;
        }
        if (obj_ind.obj_lb == 3 && !isMoving)
        {
            command_status = 2;
            print("R");
            startPoint = tracker.transform.localPosition;
            controlPoint1 = new Vector3(tracker.transform.localPosition.x, tracker.transform.localPosition.y + 0.16f, tracker.transform.localPosition.z);
            controlPoint2 = new Vector3(-pos_cylinder.y, pos_cylinder.z + 0.16f, pos_cylinder.x);
            endPoint = new Vector3(-pos_cylinder.y, pos_cylinder.z, pos_cylinder.x);
            // Start the movement when button is clicked
            if (!isMoving)
            {
                startTime = Time.time;
                isMoving = true;
                phase = 1;
            }
            reach_obj = false;
        }



        /// move to the targets
        if (obj_ind.target_lb == 4 && !isMoving)
        {
            command_status = 2;
            print("Q");
            startPoint = tracker.transform.localPosition;
            controlPoint1 = new Vector3(tracker.transform.localPosition.x, tracker.transform.localPosition.y + 0.16f, tracker.transform.localPosition.z);
            controlPoint2 = new Vector3(-target_position_close.y, target_position_close.z + 0.16f, target_position_close.x);
            endPoint = new Vector3(-target_position_close.y, target_position_close.z, target_position_close.x);
            // Start the movement when button is clicked
            if (!isMoving)
            {
                startTime = Time.time;
                isMoving = true;
                phase = 4;
            }
            reach_tg = false;
        }
        if (obj_ind.target_lb == 5 && !isMoving)
        {
            command_status = 2;
            print("W");
            startPoint = tracker.transform.localPosition;
            controlPoint1 = new Vector3(tracker.transform.localPosition.x, tracker.transform.localPosition.y + 0.16f, tracker.transform.localPosition.z);
            controlPoint2 = new Vector3(-target_position_mid.y, target_position_mid.z + 0.16f, target_position_mid.x);
            endPoint = new Vector3(-target_position_mid.y, target_position_mid.z, target_position_mid.x);
            // Start the movement when button is clicked
            if (!isMoving)
            {
                startTime = Time.time;
                isMoving = true;
                phase = 4;
            }
            reach_tg = false;
        }
        if (obj_ind.target_lb == 6 && !isMoving)
        {
            command_status = 2;
            print("E");
            startPoint = tracker.transform.localPosition;
            controlPoint1 = new Vector3(tracker.transform.localPosition.x, tracker.transform.localPosition.y + 0.16f, tracker.transform.localPosition.z);
            controlPoint2 = new Vector3(-target_position_far.y, target_position_far.z + 0.16f, target_position_far.x);
            endPoint = new Vector3(-target_position_far.y, target_position_far.z, target_position_far.x);
            // Start the movement when button is clicked
            if (!isMoving)
            {
                startTime = Time.time;
                isMoving = true;
                phase = 4;
            }
            reach_tg = false;
        }


        journeyLength = Vector3.Distance(startPoint, controlPoint1); // Set journey length to start-firstMid
        if (isMoving)
        {
            float distCovered = (Time.time - startTime) * speed;
            float fractionOfJourney = distCovered / journeyLength;

            switch (phase)
            {
                case 1:
                    // Move towards the first midpoint
                    tracker.transform.localPosition = Vector3.Lerp(startPoint, controlPoint1, fractionOfJourney);

                    // Check if the first midpoint is reached
                    if (tracker.transform.localPosition == controlPoint1)
                    {
                        phase = 2;
                        startTime = Time.time; // Reset the start time for the next part of the journey
                        journeyLength = Vector3.Distance(controlPoint1, controlPoint2); // Set journey length to firstMid-secondMid
                    }
                    break;
                case 2:
                    // Move towards the second midpoint
                    tracker.transform.localPosition = Vector3.Lerp(controlPoint1, controlPoint2, fractionOfJourney);

                    // Check if the second midpoint is reached
                    if (tracker.transform.localPosition == controlPoint2)
                    {
                        phase = 3;
                        startTime = Time.time; // Reset the start time for the next part of the journey
                        journeyLength = Vector3.Distance(controlPoint2, endPoint); // Set journey length to secondMid-end
                    }
                    break;
                case 3:
                    // Move towards the end position
                    tracker.transform.localPosition = Vector3.Lerp(controlPoint2, endPoint, fractionOfJourney);

                    // Stop moving when we reach the end position
                    if (tracker.transform.localPosition == endPoint)
                    {
                        isMoving = false;
                        Vector3 real_ee = new Vector3(reader.myArray[0], reader.myArray[1], reader.myArray[2]);
                        print((real_ee - endPoint).magnitude);
                        if ((real_ee - endPoint).magnitude <= 0.05)
                        {
                            reach_obj = true;
                            command_status = 0;
                        }
                        obj_ind.obj_lb = 0;

                    }
                    break;
                case 4:
                    // Move towards the first midpoint
                    tracker.transform.localPosition = Vector3.Lerp(startPoint, controlPoint1, fractionOfJourney);

                    // Check if the first midpoint is reached
                    if (tracker.transform.localPosition == controlPoint1)
                    {
                        phase = 5;
                        startTime = Time.time; // Reset the start time for the next part of the journey
                        journeyLength = Vector3.Distance(controlPoint1, controlPoint2); // Set journey length to firstMid-secondMid
                    }
                    break;
                case 5:
                    // Move towards the second midpoint
                    tracker.transform.localPosition = Vector3.Lerp(controlPoint1, controlPoint2, fractionOfJourney);

                    // Check if the second midpoint is reached
                    if (tracker.transform.localPosition == controlPoint2)
                    {
                        phase = 6;
                        startTime = Time.time; // Reset the start time for the next part of the journey
                        journeyLength = Vector3.Distance(controlPoint2, endPoint); // Set journey length to secondMid-end
                    }
                    break;
                case 6:
                    // Move towards the end position
                    tracker.transform.localPosition = Vector3.Lerp(controlPoint2, endPoint, fractionOfJourney);

                    // Stop moving when we reach the end position
                    if (tracker.transform.localPosition == endPoint)
                    {
                        isMoving = false;
                        Vector3 real_ee = new Vector3(reader.myArray[0], reader.myArray[1], reader.myArray[2]);
                        if ((real_ee - endPoint).magnitude <= 0.05) 
                        { 
                            reach_obj = false;
                            command_status = 0;
                        }
                        obj_ind.target_lb = 0;
                    }
                    break;
            }
        }
    }
}
