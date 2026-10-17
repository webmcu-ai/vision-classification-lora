# vision-classification-lora
vision-classification-lora


Live Demo at  https://webmcu-ai.github.io/vision-classification-lora/


Note: Must wire the LoRa module wio-sx1262, with female headers away from the B2B connector , which allows the camera and mic and sd card to use the B2B connector.


# Vision Classification with LoRa

**On-device TinyML vision classification, wireless LoRa reporting, and browser-based training and monitoring.**

Repository: https://github.com/webmcu-ai/vision-classification-lora

Live web application: https://webmcu-ai.github.io/vision-classification-lora/

Part of the [webmcu-ai](https://github.com/webmcu-ai) open-source TinyML and WebAI projects.

## 1. Overview

This project combines a small convolutional neural network (CNN), an ESP32-S3 camera, a MicroSD card, an SX1262 LoRa radio, and a browser-based training and monitoring interface.

Each device captures images, classifies them locally, and periodically transmits a compact summary of its recent classifications to other LoRa devices. A computer connected to one device can display the combined activity of the wireless network through a web browser.

The system is designed as an accessible proof of concept for understanding machine learning, embedded systems, wireless sensor networks, and privacy-respecting AI.

### What it demonstrates

- **On-device vision:** Run image classification directly on the ESP32-S3.
- **Local training:** Train a small CNN using the browser application or supported device-side workflow.
- **Dataset management:** Capture, organize, inspect, and remove training images.
- **Model persistence:** Store trained weights on the MicroSD card or compile weights into firmware.
- **LoRa networking:** Share classification summaries between nearby devices without Wi-Fi or an internet connection.
- **Browser monitoring:** View device reports, network totals, signal strength, messages, and historical activity.
- **Device configuration:** Change supported settings through the browser's USB serial connection.
- **Offline-friendly development:** Keep images and model data on the local computer and device rather than uploading them to a cloud AI service.

This is an experimental, educational system intended to make the complete pipeline inspectable and adaptable. It is not a certified communications product or a production security system.

## 2. System Architecture

The project contains three main parts.

### A. ESP32-S3 firmware

File: `firmware.ino`

The firmware controls the camera, SD card, neural network, optional OLED, and LoRa radio.

Each board independently:

1. Captures camera images.
2. Collects labelled examples into class folders on the SD card.
3. Trains or loads a CNN model using the supported firmware workflow.
4. Runs inference on new images.
5. Counts predictions that meet a configurable confidence threshold.
6. Periodically transmits a summary over LoRa.
7. Receives and displays summaries and text messages from other boards.
8. Reports device information and debugging data over USB serial.

### B. Browser application

File: `index.html`

The single-page web application provides tools for:

- Loading a dataset from an SD-card folder or ZIP archive.
- Capturing images using the computer's webcam.
- Organizing class folders and reviewing images.
- Training and evaluating the browser-side CNN.
- Viewing training loss, accuracy, and classification results.
- Exporting trained model weights and configuration files.
- Connecting to an ESP32-S3 through Web Serial.
- Displaying device camera and inference information.
- Monitoring LoRa devices and combining their reports.
- Sending LoRa text messages through the connected board.
- Recording, graphing, importing, and exporting network history.

The browser does not need to run the CNN on behalf of every remote device. Each remote ESP32 performs its own inference and sends its results over LoRa.

### C. LoRa network

The wireless network uses Seeed Studio Wio-SX1262 hardware with the SX1262 radio.

Each participating device needs compatible firmware, matching radio settings, and a unique device name.

The devices exchange compact reports containing classification counts, as well as optional text messages. The network does not require a Wi-Fi access point, router, cloud service, or internet connection.

The browser needs a USB serial connection to one participating device to observe the traffic that device receives.

## 3. Hardware

### Recommended hardware

| Component | Purpose |
|---|---|
| Seeed Studio XIAO ESP32-S3 Sense | Main microcontroller and camera |
| XIAO ML Kit expansion board | MicroSD storage and optional OLED/IMU hardware |
| Wio-SX1262 LoRa module | Long-range, low-data-rate wireless communication |
| MicroSD card | Training images, model weights, and configuration |
| USB-C data cable | Firmware upload and browser serial connection |
| Suitable LoRa antenna | Radio transmission and reception |

The camera, SD card, and LoRa module must be connected so that their pin assignments do not conflict.

### Important: The LoRa Module's B2B Connector Is Not Used

The Wio-SX1262 LoRa module is connected through the regular XIAO pins using female headers fitted to the LoRa module. Its B2B connector is deliberately left unused in this project.

The reason is that the XIAO ESP32-S3 Sense needs its B2B connector for the camera. The external-header arrangement allows the camera and LoRa module to operate together without using the LoRa module's B2B connector.

This is also an intentional modular hardware design decision:

- **Camera compatibility:** The XIAO's B2B connector remains available for the camera.
- **Regular pin connections:** The LoRa module communicates through the normal XIAO pins using female headers.
- **Unobstructed B2B connector:** The LoRa module's B2B connector is not blocked by the header arrangement.
- **Future flexibility:** The LoRa module's B2B connector can still be used in a different project where that connection method is appropriate.
- **Modularity:** The same LoRa module can support different hardware configurations without requiring its B2B connector to be the default connection method.

The female headers are a physical connection choice. They do not change the electrical requirements of the radio. The external LoRa pin assignments must match the firmware, and the wiring must avoid conflicts with the camera, SD card, USB interface, and optional peripherals.

Fit the correct antenna before powering the LoRa module.

## 4. LoRa Wiring

The firmware documents the following external SX1262 connections:

| Wio-SX1262 signal | XIAO pin | ESP32-S3 GPIO |
|---|---|---:|
| SCK | D8 | GPIO 7 |
| MISO | D9 | GPIO 8 |
| MOSI | D10 | GPIO 9 |
| NSS / CS | D3 | GPIO 4 |
| RESET | D2 | GPIO 3 |
| BUSY | D1 | GPIO 2 |
| DIO1 | D0 | GPIO 1 |
| Power | 3V3 | 3.3 V |
| Ground | GND | GND |

The SPI clock, MISO, and MOSI signals are shared with the SD-card SPI bus. The SD card uses GPIO 21 for chip select.

This arrangement allows the camera and SD-card expansion board to remain connected while the LoRa module uses external connections through the regular XIAO pins.

D0 is also associated with the optional touch input. Because D0 is used for LoRa DIO1, touch input is disabled by default. Do not enable it without resolving the pin conflict.

Check the firmware definitions before changing the wiring. A pin assignment that works for the LoRa module alone may conflict with the camera, SD card, USB serial interface, or optional display.

## 5. Software Requirements

### Browser

Use a recent version of Google Chrome or Microsoft Edge.

The application uses browser capabilities including:

- Web Serial for communication with the ESP32-S3.
- File and directory selection for dataset management.
- Webcam access for image capture.
- JavaScript for training, evaluation, visualization, and history management.

Web Serial is not supported in every browser. Webcam access may require a secure context, such as the hosted HTTPS application or localhost.

### Firmware Development

The firmware is written for the Arduino ESP32-S3 environment.

The source identifies these principal libraries:

- Arduino ESP32 camera support (`esp_camera.h`).
- `U8g2` by olikraus for the optional OLED.
- `RadioLib` by jgromes for the SX1262 radio.
- Arduino SD, SPI, Wire, and Preferences facilities.

Enable **USB CDC On Boot** and configure **OPI PSRAM** in the Arduino IDE when compiling for the documented XIAO ESP32-S3 target.

The firmware also contains configuration for a PlatformIO build. If using PlatformIO, check the repository configuration and ensure PSRAM and USB CDC settings match the firmware requirements.

## 6. Getting Started

### Step 1 — Open the Browser Application

Open the [live application](https://webmcu-ai.github.io/vision-classification-lora/).

Alternatively, clone the repository:

```bash
git clone https://github.com/webmcu-ai/vision-classification-lora.git
