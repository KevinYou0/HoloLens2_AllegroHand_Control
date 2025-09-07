using System.Collections;
using System.IO.Ports;
using UnityEngine;

public class SerialLuminanceReader : MonoBehaviour
{
    private SerialPort serialPort;
    public string portName = "COM3"; // Change this to your Arduino's port name
    public int baudRate = 9600;

    public delegate void LuminanceReceived(float luminanceValue);
    public static event LuminanceReceived OnLuminanceReceived;

    public float luminanceValue;

    private void Start()
    {
        string[] ports = SerialPort.GetPortNames();

        Debug.Log("The following serial ports were found:");

        // Display each port name to the console.
        foreach (string port in ports)
        {
            Debug.Log(port);
        }

        serialPort = new SerialPort(portName, baudRate);
        serialPort.Open();
        StartCoroutine(ReadSerialData());
    }

    private IEnumerator ReadSerialData()
    {
        while (true)
        {
            if (serialPort.IsOpen && serialPort.BytesToRead > 0)
            {
                try
                {
                    string data = serialPort.ReadLine();
                    if (data.StartsWith("LUM:"))
                    {
                        luminanceValue = float.Parse(data.Substring(4));
                        //Debug.Log("Luminance: " + luminanceValue);
                        // Use luminanceValue in your game logic here
                        OnLuminanceReceived?.Invoke(luminanceValue);  // Raise the event with the luminance value
                    }
                }
                catch (System.Exception e)
                {
                    Debug.LogError("Error reading from serial port: " + e.Message);
                }
            }
            yield return new WaitForSeconds(0.1f); // Wait for 100ms before the next read
        }
    }

    private void OnDestroy()
    {
        if (serialPort != null && serialPort.IsOpen)
        {
            serialPort.Close();
        }
    }
}