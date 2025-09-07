using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using RosSharp.RosBridgeClient;
using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.Utilities;

public class CalibPos_visual_fake : MonoBehaviour
{
    public Transform localTran;
    public GameObject ee_hand;
    public GameObject tracker;
    public JoySubscriber_vs reader;
    public bool reach_obj = false;
    public bool reach_tg = false; 
    public Vector3 ee_hand_pos;
    public GameObject dotPrefab;
    private GameObject currentDotInstance;
    
    List<string> target_tag = new List<string>() { "star", "cylinder", "round", "far", "mid", "close" };

    Vector3 startPoint;
    Vector3 controlPoint1;
    Vector3 controlPoint2;
    Vector3 endPoint;

    // this is in the robot arm system
    Vector3 pos_cylinder = new Vector3(0.52094144f, -0.11900946f, 0.294116638f);
    Vector3 pos_star = new Vector3(0.52141990f, 0.00945116f, 0.29992781f);
    Vector3 pos_small_cylinder = new Vector3(0.52100867f, 0.12295886f, 0.29849644f);

    Vector3 target_position_close = new Vector3(0.44065475f, 0.28345066f, 0.37329465f);
    Vector3 target_position_mid = new Vector3(0.56062072f, 0.28442630f, 0.35793865f);
    Vector3 target_position_far = new Vector3(0.67626953f, 0.27326885f, 0.37563881f);

    Vector3 temp_obj;

    private bool isMoving = false;
    private float speed = 0.1f;
    private float startTime;
    private float journeyLength;
    private int phase = 0;
    private float stareTimer = 0f;
    private int obj_lb = 0;
    private int target_lb = 0;
    float activate_gripper = 10f;
    bool on_mission = false;

    public int move_count;
    public int target_count;
    public int command_status = 0;
    public int IsPinching = 0;
    public bool IsMoving = false;
    MixedRealityPose pose_temp;
    string hitObjectName;

    private bool CheckListForKeywords(string inputString, List<string> keywords)
    {
        foreach (string keyword in keywords)
        {
            if (inputString.Contains(keyword))
            {
                return true; // A keyword was found in this list
            }
        }
        return false; // No keywords found in this list
    }

    void Start()
    {
        GameObject star = GameObject.Find("star.Shrink");
        star.transform.localPosition = new Vector3(-pos_star.y, pos_star.z, pos_star.x);
        GameObject Cyl_big = GameObject.Find("Cyl_big.Shrink");
        Cyl_big.transform.localPosition = new Vector3(-pos_cylinder.y, pos_cylinder.z, pos_cylinder.x);
        GameObject Cyl_sml = GameObject.Find("Cyl_small.Shrink");
        Cyl_sml.transform.localPosition = new Vector3(-pos_small_cylinder.y, pos_small_cylinder.z, pos_small_cylinder.x);
    }

    IEnumerator YourFunctionAtEndOfFrame()
    {
        yield return new WaitForEndOfFrame();
        on_mission = false;
    }

    // Update is called once per frame
    void Update()
    {

        if (Input.GetKeyDown(KeyCode.D))
        {
            IsMoving = true;
        }
        else
        {
            IsMoving = false;
        }

        GameObject ThumbTip = GameObject.Find("ThumbTip Proxy Transform");
        GameObject IndexTip = GameObject.Find("IndexTip Proxy Transform");
        if (ThumbTip != null && IndexTip != null)
        {
            //if ((IndexTip.transform.position - ThumbTip.transform.position).magnitude < 0.05f)
            //{
            //    IsPinching = 1;
            //}
            //if ((IndexTip.transform.position - ThumbTip.transform.position).magnitude > 0.05f)
            //{
            //    IsPinching = 2;
            //}
        }

        if (currentDotInstance == null)
        {
            currentDotInstance = Instantiate(dotPrefab);
        }

        //if (CheckListForKeywords(hitObjectName, target_tag))
        //{
        //    Debug.Log("find object: " + hitObjectName);
        //    stareTimer += Time.deltaTime;
        //}
        //else
        //{
        //    on_mission = false;
        //}

        if (Input.GetKeyDown(KeyCode.L)) 
        {
            hitObjectName = "round";
        }
        if (Input.GetKeyDown(KeyCode.M))
        {
            hitObjectName = "star";
        }
        if (Input.GetKeyDown(KeyCode.R))
        {
            hitObjectName = "cylinder";
        }

        if (Input.GetKeyDown(KeyCode.Q))
        {
            hitObjectName = "close";
        }
        if (Input.GetKeyDown(KeyCode.A))
        {
            hitObjectName = "star";
        }
        if (Input.GetKeyDown(KeyCode.Z))
        {
            hitObjectName = "cylinder";
        }

        // Check if the object has been stared at for 1 second
        if (IsMoving && !on_mission)
        {
            StartCoroutine(YourFunctionAtEndOfFrame());
            print("move once");
            on_mission = true;
            //stareTimer = 0;

            if (hitObjectName == "round" || hitObjectName == "star" || hitObjectName == "cylinder") 
            {
                IsPinching = 2;
                move_count += 1;
                obj_lb = 1;

                if (hitObjectName == "round") 
                {
                    temp_obj = pos_small_cylinder;
                }
                if (hitObjectName == "mid")
                {
                    temp_obj = pos_star;
                }
                if (hitObjectName == "far")
                {
                    temp_obj = pos_cylinder;
                }

            }
            else 
            { 
                obj_lb = 0; 
            };

            if (hitObjectName == "close" || hitObjectName == "mid" || hitObjectName == "far") 
            {
                IsPinching = 1;
                target_count += 1;
                target_lb = 1;
                if (hitObjectName == "close")
                {
                    temp_obj = target_position_close;
                }
                if (hitObjectName == "mid")
                {
                    temp_obj = target_position_mid;
                }
                if (hitObjectName == "far")
                {
                    temp_obj = target_position_far;
                }
            }
            else
            { 
                target_lb = 0; 
            };
        }



        if (Input.GetKeyDown(KeyCode.C))
        {
            print("C");
            tracker.transform.localPosition = new Vector3(reader.myArray[0], reader.myArray[1], reader.myArray[2]);
        }

        /// move to the objects
        if (obj_lb == 1 && !isMoving) 
        {
            startPoint = tracker.transform.localPosition;
            controlPoint1 = new Vector3(tracker.transform.localPosition.x, tracker.transform.localPosition.y + 0.16f, tracker.transform.localPosition.z);
            controlPoint2 = new Vector3(-temp_obj.y, temp_obj.z + 0.16f, temp_obj.x);
            endPoint = new Vector3(-temp_obj.y, temp_obj.z, temp_obj.x);
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
        if (target_lb == 1 && !isMoving)
        {
            startPoint = tracker.transform.localPosition;
            controlPoint1 = new Vector3(tracker.transform.localPosition.x, tracker.transform.localPosition.y + 0.16f, tracker.transform.localPosition.z);
            controlPoint2 = new Vector3(-temp_obj.y, temp_obj.z + 0.16f, temp_obj.x);
            endPoint = new Vector3(-temp_obj.y, temp_obj.z, temp_obj.x);
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
                        }
                        obj_lb = 0;
                        IsPinching = 1;

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
                            reach_tg = false;
                        }
                        target_lb = 0;
                        IsPinching = 2;
                    }
                    break;
            }
        }
    }
}
