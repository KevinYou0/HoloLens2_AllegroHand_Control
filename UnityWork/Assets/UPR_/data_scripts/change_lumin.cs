using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class change_lumin : MonoBehaviour
{
    public GameObject time_set;
    private float change_time;
    private float time;
    private float lumin_value;
    private Light point_light;
    private float delta_lumin;
    public float max_lumin;
    // Start is called before the first frame update
    void Start()
    {
        point_light = this.GetComponent<Light>();
        change_time = time_set.GetComponent<ChangeColor>().change_time;
        delta_lumin = max_lumin / (change_time * 2);
    }

    // Update is called once per frame
    void Update()
    {
        time += Time.deltaTime;
        change_lumin_fun();
    }
    void change_lumin_fun()
    {
        if (time < change_time)
        {
            lumin_value = 0f;
            point_light.intensity = lumin_value;
        }
        if (time > change_time && time < change_time * 3)
        {
            lumin_value = 0f + delta_lumin * (time - change_time);
            point_light.intensity = lumin_value;
        }
        if (time > change_time * 3 && time < change_time * 4)
        {
            lumin_value = max_lumin;
            point_light.intensity = lumin_value;
        }
        if (time > change_time * 4 && time < change_time * 6)
        {
            lumin_value = max_lumin - delta_lumin * (time - change_time * 4);
            point_light.intensity = lumin_value;
        }
        if (time > change_time * 6 && time < change_time * 7)
        {
            lumin_value = 0f;
            point_light.intensity = lumin_value;
        }
        //print(lumin_value);
    }
}
