using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class CogLoad : MonoBehaviour
{
    public int windowLength = 20;
    public int startLength = 300;
    private int window_i = 0;
    private List<float> pupil_array = new List<float>();

    private float std;

    private float lum_new_sum = 0;
    private float lum_new_average = 0;
    private float pupil_sum = 0;
    private float pupil_average = 0;

    private float pupil_new_average = 0;

    private float trend = 0;
    private float trend_sum = 0;
    private float cogload_new = 0;
    private float cogload_old = 0;

    private float adjustLum = 0;
    private bool start = false;

    private float camera_sum;
    private float speed;
    private float time;
    private float blink_new = 0f;
    private float blink_old = 0f;
    private int blink_eye = 0;
    private int blink_i = 0;
    private float pupil_avg = 0;

    private Vector3 coeff_left;
    private Vector3 coeff_right;
    private float default_blink_rate;
    //private TextAsset file;
    public float multi_lumin=1;

    // Copy this to DataRecCsv.cs
    //------------------------------------------------------
    //public float getRP()
    //{
    //    return pd_right;
    //}
    //public float getLP()
    //{
    //    return pd_left;
    //}
    //public float getLum()
    //{
    //    return (float)lum;
    //}
    // -----------------------------------------------------

    // Start is called before the first frame update
    void Start()
    {

        coeff_left = new Vector3(2.451072052f, -5.47037207f, 2.49154129f);
        coeff_right = new Vector3(2.489320955f, -4.62467234f, 2.540344354f);
        default_blink_rate = 10;
        
    }

    // Update is called once per frame
    void Update()
    {

        if (start == false)
        {
            if (window_i < startLength)
            {
                if (this.GetComponent<CsvAssistedAutonomy>().GetPupil() > 0)
                {
                    lum_new_sum += this.GetComponent<CsvAssistedAutonomy>().GetLum() * multi_lumin;
                    pupil_array.Add(this.GetComponent<CsvAssistedAutonomy>().GetPupil());
                    window_i++;
                    //Debug.Log(window_i);
                }
            }
            if (window_i >= startLength)
            {
                preprocessData();
                //adjustRightLum = pupil_right_average - lum_new_average_zty;
                //adjustLeftLum = pupil_left_average - lum_new_average_zty;
                pupil_avg = pupil_average;
                start = true;
                window_i = 0;
                lum_new_sum = 0;
            }
        }
        if (start == true)
        {
            if (blink_i < 500)
            {
                // blink count
                blink_new = this.GetComponent<CsvAssistedAutonomy>().GetPupil();
                if (blink_old < 0 & blink_new > 0) { blink_eye++; }
                blink_old = blink_new;
                //Debug.Log("blink_eye: " + blink_eye);
                blink_i++;
            }
            else
            {
                blink_i = 0;
                blink_eye = 0;
            }
            if (window_i < windowLength)
            {
                if (this.GetComponent<CsvAssistedAutonomy>().GetPupil() > 0)
                {

                    lum_new_sum += this.GetComponent<CsvAssistedAutonomy>().GetLum() * multi_lumin;
                    pupil_array.Add(this.GetComponent<CsvAssistedAutonomy>().GetPupil());
                    time += Time.deltaTime;
                    window_i++;
                    // output for other 49 frames
                    trend = 0;
                }

            }
            if (window_i >= windowLength)
            {
                preprocessData();
                speed = (camera_sum) / time;
                trend = trend + cogload_new - cogload_old;
                cogload_old = cogload_new;
                trend_sum += trend;

                window_i = 0;
                lum_new_sum = 0;
                camera_sum = 0;
                time = 0;
            }
        }


    }


    public float normLum(float x)
    {
        //float y = (float)(-3 * Mathf.Pow(10, -6) * Mathf.Pow(x, 3) + 0.00036 * Mathf.Pow(x, 2) - 1.5827 * x + 315.66) / 10;
        // zty left model
        //float y = (float)(2.5202 * Mathf.Pow(10,-3) * Mathf.Pow(x, 2) - 1.5318 * Mathf.Pow(10, -1) * Mathf.Pow(x, 1) + 4.8655 * Mathf.Pow(x, 0));
        // EXP model //
        float y = coeff_left[0]*Mathf.Exp(coeff_left[1]*x) + coeff_left[2];
        return y;
        // Excel =(-3*POWER(10,-6)*POWER(AF2*255,3)+0.00036*POWER(AF2*255,2)-1.5827*AF2*255+315.66)/10 - 27
    }

    private void StandardDeviation(List<float> arrayData, out float total_std, out float sum, out float avg)
    {
        sum = 0f;
        for (int i = 0; i < arrayData.Count; i++) {sum += arrayData[i]; }
        avg = sum / arrayData.Count;

        float total_sum = 0f;
        for (int i = 0; i < arrayData.Count; i++) { total_sum += ((arrayData[i] - avg) * (arrayData[i] - avg)); }
        total_std = Mathf.Sqrt(total_sum / arrayData.Count);

    }

    private void preprocessData()
    {
        StandardDeviation(pupil_array, out std, out pupil_sum, out pupil_average);
        lum_new_average = lum_new_sum / pupil_array.Count;

        pupil_sum = 0;
        for (int i = 0; i < pupil_array.Count; i++)
        {
            pupil_sum += pupil_array[i];
        }
        pupil_average = pupil_sum / pupil_array.Count;

        lum_new_average = normLum(lum_new_average);
        //pupil_new_average = (pupil_left_average + pupil_right_average - adjustRightLum - adjustLeftLum) / 2;
        pupil_new_average = (pupil_average - adjustLum);
        cogload_new = pupil_new_average - lum_new_average*1.0f;
        pupil_array.Clear();
    }

    public float getTrend()
    {
        return trend;
    }
    public float getTrendsum()
    {
        return trend_sum;
    }
 
    public float getNewCog()
    {
        return cogload_new;
    }
    public float getOldCog()
    {
        return cogload_old;
    }
    public float getSpeed()
    {
        return speed;
    }
    public float getPupilAvg()
    {
        return pupil_avg;
    }
    public float getBlinkRate()
    {
        return default_blink_rate;
    }
}
