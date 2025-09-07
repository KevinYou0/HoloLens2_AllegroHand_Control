using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class RecordLumin : MonoBehaviour
{
    public string fileName;
    private string filePath;
    private StreamWriter writer;

    public GameObject Lumin_scene;
    public GameObject Lumin_sensor;
    public GameObject Pupil;

    // Start is called before the first frame update
    void Start()
    {
        filePath = Application.dataPath + "/Resources/Lumin_calib/" + fileName + ".csv";
        writer = new StreamWriter(filePath);
        WriteCSVHeader();
    }

    // Update is called once per frame
    void Update()
    {
        
        string str = "";
        str = str + Time.time + ",";
        str = str + Lumin_scene.GetComponent<Luminosity>().lum + ",";
        str = str + Lumin_sensor.GetComponent<SerialLuminanceReader>().luminanceValue + ",";
        str = str + Pupil.GetComponent<PupilLabs.Demos.PupilDataDemo>().pupil_size + ",";
        str = str + Pupil.GetComponent<PupilLabs.Demos.PupilDataDemo>().pupil_ID + ",";
        //str = str + this.GetComponent<CogLoad>().getNewCog() + ",";
        //str = str + this.GetComponent<CogLoad>().getTrend() + ",";
        //str = str + this.GetComponent<CogLoad>().getTrendsum() + ",";
        //str = str + this.GetComponent<CogLoad>().getPupilAvg() + ",";



        writer.WriteLine(str);
    }

    private void WriteCSVHeader()
    {
        //string[] header = new string[8] {"time", "Lumin", "PupilSize", "PupilID", "CogLoad", "Trend", "TrendSum", "PupilAvg"};
        string[] header = new string[5] {"time", "Lumin_scene", "Lumin_measured", "PupilSize", "PupilID"};
        string str = "";

        for (int i = 0; i < header.Length; i++)
        {
            str = str + header[i] + ",";
        }

        writer.WriteLine(str);
        print("writer open");
    }

    void OnApplicationQuit()
    {
        writer.Flush();
        writer.Close();
        print("close CSV writer");
    }

    public float GetLum()
    {
        return (float)Lumin_scene.GetComponent<Luminosity>().lum;
    }

    public float GetPupil()
    {
        return (float)Pupil.GetComponent<PupilLabs.Demos.PupilDataDemo>().pupil_size;
    }


}
