// ======================================================
// XIAO ESP32S3 SENSE (OR XIAO ML KIT): VISION CNN + LORA NETWORK
// firmware-lora-v007  (works with index-lora-v006.html)
// ======================================================
// Small image collection, training and inference for education and proof of concept.
//  - Images are collected per class on the SD card, a small CNN is trained on the board, then it classifies the camera.
//  - The SD card stores the images in class folders and the weights as header/myWeights.bin and as a .h text char array.
//  - Class labels are read from /header/config.json at boot when the number of labels equals NUM_CLASSES.
//  - Serial Monitor and OLED output. The OLED is optional: it is looked for at boot (MY_OLED_ADDRESS) and, if nothing
//    answers on D4/D5, all screen output is skipped so a board without a display runs at full speed.
//
// LoRa network
//  - While inferring, every MY_DEFAULT_REPORT_SEC seconds (default 30) the board sends a short LoRa summary:
//      S,<name>,<seq>,<periodSec>,<frames>,<c0>,<c1>,...     (frames per class with confidence >= @conf, the rest are "unsure")
//  - Every board prints the summaries it hears, and its own, with a time stamp and a description in plain words.
//  - The board connected to the web page by Web Serial also prints machine lines for the page:
//      @LORA <rssi|self> <snr> <packet>       summaries
//      @LORA-MSG <rssi|self> <snr> <text>     chat messages
//      @LORA-INFO name=... ch=... classes=... board settings (only while the page is connected, or after @pageinfo)
//    The page sums the summaries of all boards over the last few minutes.
//  - Messages: type  >your text  and Enter (works in every mode), or open the "LoRa msg" menu item.
//
// Main menu: one item per class (collect images), Train, Infer, LoRa msg.
//   Serial Monitor: t = next, l = select, or press the item's digit. The A0 touch pad works when MY_TOUCH_ENABLED is 1.
//
// Commands (type, then Enter; @help lists them with their current values)
//   >text   @name @report @channel @conf @autostart @autodelay @print @touch @encrypt @seed
//   @menu @status @time @reset @help
//   Every setting has its default in USER SETTINGS below. A value changed by command is saved in flash and wins over
//   the default until @reset.
//
// Arduino IDE: Board "XIAO_ESP32S3", Tools -> USB CDC On Boot: Enabled, Tools -> PSRAM: OPI PSRAM.
// Libraries: U8g2 (olikraus) and RadioLib (jgromes).
//
// By Jeremy Ellis
// With free tier assistance from: Claude (code overview), ChatGPT (Critique), Gemini (Research) and Copilot (Alternate)
// Use at your own risk!
// MIT license
//
// Github Profile https://github.com/hpssjellis
// LinkedIn https://www.linkedin.com/in/jeremy-ellis-4237a9bb/
//
// For platformio you need the libraries declared in the platformio.ini file and OPI PSRAM set
// lib_deps =  olikraus/U8g2 @ ^2.35.30
//             jgromes/RadioLib
// ; Overriding defaults to enable OPI PSRAM
// build_flags = 
//    -DBOARD_HAS_PSRAM
//    -DARDUINO_USB_CDC_ON_BOOT=1
// board_build.arduino.memory_type = qio_opi
// board_build.flash_mode = qio
// board_upload.flash_size = 8MB
//


// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 0: CORE SYSTEM (ALWAYS INCLUDED)                                   ██
// ██  Headers, Defines, Pins, Globals, Memory, Weights, Setup, Loop           ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████


// ============================================================================
//  USER SETTINGS  -  the defaults you are most likely to change.  Edit, then re-upload.
//  Every value below also has an @command (shown in @help with its current value). A value you change with a command is
//  SAVED IN THE BOARD'S FLASH and wins over these defaults. @help shows which values are saved, @reset goes back to these.
// ============================================================================
#define MY_DEFAULT_NAME        "device-a01"   // board name sent in every LoRa report; make it unique per board
#define MY_DEFAULT_REPORT_SEC  30             // seconds between LoRa summaries (5..3600)
#define MY_DEFAULT_CHANNEL     0              // 0..120, frequency = 915.0 MHz + channel x 0.1 MHz; all boards must match
#define MY_DEFAULT_MIN_CONF    60             // % confidence a frame needs to be counted as a class (else "unsure")
#define MY_DEFAULT_AUTO_START  1              // after a power cycle: 1 = start inference by itself (needs trained weights), 0 = show the menu
#define MY_DEFAULT_AUTO_DELAY_S 5             // countdown (seconds) before auto-start; send t, l or a digit to stay in the menu
#define MY_DEFAULT_ENCRYPT     0              // 1 = LoRa text scrambled with the seed below (hides text from casual listeners only)
#define MY_DEFAULT_SEED        "maker100"     // encryption seed, must match on all boards
#define MY_TOUCH_ENABLED       0              // 1 = use the A0 (D0) touch pad. Must stay 0 while LORA_DIO1 is wired to D0
#define MY_OLED_ADDRESS        0x3C           // 7-bit I2C address of the OLED, probed at boot. Nothing answering = no display, screen output is skipped
#define MY_DEFAULT_TOUCH_EXIT  1              // with the touch pad on: 0 = ignore it while inferring (headless units)
#define MY_DEFAULT_PRINT_EVERY 1              // print the per-frame "Current Pred" line every N frames (1 = every frame, 10 = quieter monitor)
// ============================================================================

// Optional: uncomment AFTER copying myWeights.h from the SD card to your sketch folder:
// Priority order: SD weights > baked-in weights > random He-init
//////////////////////////////////////IMPORTANT/////////////////////////////////////////////////
//#define USE_BAKED_WEIGHTS

#ifdef USE_BAKED_WEIGHTS
  #include "myWeights.h"
#endif

#include "esp_camera.h"
#include "img_converters.h"
#include "FS.h"
#include "SD.h"
#include "SPI.h"
#include <vector>
#include <algorithm>
#include <U8g2lib.h>
#include <Wire.h>
#include "mbedtls/base64.h"   // base64 for the Web Serial debug frames
#include <RadioLib.h>        // SX1262 LoRa radio
#include <Preferences.h>     // settings kept in flash

// OLED on I2C (D4 = SDA, D5 = SCL). With no display connected every page update would wait for an answer that never
// comes (I2C timeouts, slow frames), so firstPage()/nextPage() do nothing when 'present' is false.
class MyOled : public U8G2_SSD1306_72X40_ER_1_HW_I2C {
 public:
  bool present = true;
  MyOled() : U8G2_SSD1306_72X40_ER_1_HW_I2C(U8G2_R2, U8X8_PIN_NONE) {}
  void firstPage() { if (present) U8G2_SSD1306_72X40_ER_1_HW_I2C::firstPage(); }
  uint8_t nextPage() { return present ? U8G2_SSD1306_72X40_ER_1_HW_I2C::nextPage() : 0; }
};
MyOled u8g2;

bool myOledProbe() {
  Wire.begin();                                   // default SDA/SCL pins (D4/D5)
  Wire.beginTransmission(MY_OLED_ADDRESS);
  return Wire.endTransmission() == 0;             // 0 = the display answered
}

// ======================================================
// CONFIGURATION & ML HYPERPARAMETERS
// ======================================================


#define NUM_CLASSES 3

String myClassLabels[NUM_CLASSES] = {"0Blank", "1Cup", "2Pen"};

const int myTotalItems = NUM_CLASSES + 3;      // menu items: one per class, then Train, Infer, LoRa msg

// ============================================================================
// LORA DECLARATIONS: pins, constants, global variables and function prototypes.
// The Arduino IDE needs everything declared before it is used, so all of it lives here, near the top.
// ============================================================================
// SX1262 wiring to the XIAO ESP32S3 Sense header pins. SCK, MISO and MOSI are shared with the SD card (chip select GPIO21).
//   SX1262 pin   XIAO pin   GPIO
//   SCK          D8         7
//   MISO         D9         8
//   MOSI         D10        9
//   NSS          D3         4
//   RST          D2         3
//   BUSY         D1         2
//   DIO1         D0         1    (also the A0 touch pad pin, so the touch pad is off: see MY_TOUCH_ENABLED)
//   3V3 and GND  3V3, GND
// The camera uses the XIAO B2B connector, so a Wio-SX1262 stacked on that connector cannot be used with the camera.
// Fit the antenna BEFORE powering the module.
#define LORA_SCK    D8
#define LORA_MISO   D9
#define LORA_MOSI   D10
#define LORA_NSS    D3
#define LORA_RST    D2
#define LORA_BUSY   D1
#define LORA_DIO1   D0   // Tools --> USB CDC On Boot needs to be enabled (Serial uses the USB port)
#define MY_SD_CS    21   // SD card chip select (B2B connector)

#define MY_LORA_RXEN   -1      // RF switch pins if the module has them (Wio-SX1262 uses RXEN), -1 = not used
#define MY_LORA_TXEN   -1
#define MY_LORA_TCXO_V 1.8f    // TCXO voltage, 0 for a module without a TCXO
#define MY_LORA_DIO2_RF true   // DIO2 switches the antenna between transmit and receive
#define MY_LORA_MAX    100     // longest packet, name and text included

#if MY_TOUCH_ENABLED
  #define MY_LEAVE_HINT "t, l, @menu, or hold the touch pad."
#else
  #define MY_LEAVE_HINT "t, l or @menu."
#endif

SX1262 myRadio = new Module(LORA_NSS, LORA_DIO1, LORA_RST, LORA_BUSY);
Preferences myPrefs;
volatile bool myLoraFlag = false;
bool myLoraOk = false, myLoraTxBusy = false;
unsigned long myLoraTxStart = 0;
char myLoraName[24] = MY_DEFAULT_NAME;
int myLoraChannel = MY_DEFAULT_CHANNEL;
bool myLoraEncrypt = MY_DEFAULT_ENCRYPT;
char myLoraSeed[32] = MY_DEFAULT_SEED;
int myLoraReportSec = MY_DEFAULT_REPORT_SEC;
int myLoraMinConf = MY_DEFAULT_MIN_CONF;
bool myAutoStart = MY_DEFAULT_AUTO_START;
int myAutoDelay = MY_DEFAULT_AUTO_DELAY_S;
int myPrintEvery = MY_DEFAULT_PRINT_EVERY;
bool myTouchExit = MY_DEFAULT_TOUCH_EXIT;
bool myWantMenu = false;     // set by @menu: leaves the LoRa msg screen
unsigned long myLoraSeq = 0, myLoraTxN = 0, myLoraRxN = 0, myLoraErrN = 0;
uint16_t myLoraCounts[NUM_CLASSES];
uint32_t myLoraFrames = 0;
unsigned long myLoraWinStart = 0, myLoraNext = 0, myLoraInfoLast = 0;

// Function prototypes (the functions are defined in the LORA section further down)
void IRAM_ATTR myLoraIsr();
float myLoraFreq();
void myLoraCipher(char* b, bool enc);
void myLoraSave();
void myLoraLoad();
void myLoraPrintInfo(bool force);
void myLoraResetDefaults();
void myLoraBegin();
void myLoraCount(int pred, float p);
void myLoraStamp(char* out, size_t n);
bool myLoraExplain(const char* pkt, char* name, size_t nn, char* out, size_t n);
bool myLoraTransmit(char* pkt);
bool myLoraSendText(const char* text);
void myLoraReport();
void myLoraService();
void myLoraHelp();
void myLoraStatus();
void myLoraCommand(char* l);
void myLoraDrawChat();
void myActionLoraChat();
bool myHandleDebugChar(char c);
void myDrawMenu();
void myResetMenuState();


float LEARNING_RATE = 0.0003;
int BATCH_SIZE = 6;
int TARGET_EPOCHS = 20;
int VALIDATION_IMAGES = 3;  // last N images per class are held out for validation (0 = disabled)

const int myThresholdPress = 1100;
const int myThresholdRelease = 900;
//const unsigned long myScreenTimeout = 300000; // not used presently


// ======================================================
// CAMERA IMAGE SETTINGS
// The web trainer page shows images upright, mirrored left-right. To match it the sensor is
// mirrored AND flipped vertically. Brightness and AE level range from -2 to 2 (0 = sensor default).
// If images are still darker than the web page, raise MY_CAM_BRIGHTNESS or MY_CAM_AE_LEVEL to 2.
// If they wash out, lower them. Existing images on the SD card are NOT changed by these settings.
// ======================================================
#define MY_CAM_HMIRROR        1
#define MY_CAM_VFLIP          1
#define MY_CAM_BRIGHTNESS     1
#define MY_CAM_AE_LEVEL       1
#define MY_CAM_WARMUP_FRAMES  5   // frames discarded after start so auto exposure can settle





// ======================================================
// TOUCH INPUT: tap = next, 3+ quick taps ("long press") = select
// ======================================================
struct TouchState {
  bool isTouching = false;
  int tapCount = 0;
  unsigned long firstTapTime = 0;
  unsigned long lastReleaseTime = 0;
  unsigned long lastCheckTime = 0;  // when the pad was last read
  const unsigned long tapWindow = 800;        // ms in which taps are counted (long enough while the CNN is busy)
  const int longPressTaps = 3;                // 3+ taps = long press
  const unsigned long debounceDelay = 50;     // debounce time
};


TouchState myTouch;








// SYSTEM LOGIC VARIABLES
unsigned long myLastActivityTime = 0; 
unsigned long myLastTapTime = 0;
const int myTapCooldown = 250;
int myMenuIndex = 1;
bool myIsSelected = false;
bool myWeightsTrained = false; 

// XIAO ESP32-S3 Camera Pins
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     10
#define SIOD_GPIO_NUM     40
#define SIOC_GPIO_NUM     39
#define Y9_GPIO_NUM       48
#define Y8_GPIO_NUM       11
#define Y7_GPIO_NUM       12
#define Y6_GPIO_NUM       14
#define Y5_GPIO_NUM       16
#define Y4_GPIO_NUM       18
#define Y3_GPIO_NUM       17
#define Y2_GPIO_NUM       15
#define VSYNC_GPIO_NUM    38
#define HREF_GPIO_NUM     47
#define PCLK_GPIO_NUM     13

// ======================================================
// CONFIGURABLE INPUT RESOLUTION
// Square and EVEN (the 2x2 max pool needs INPUT_SIZE-2 to be even). Web page supports 16..128.
// Images on the SD card are always 240x240; they are resampled to INPUT_SIZE when loaded.
// Larger sizes train and infer much more slowly (cost grows with the square of INPUT_SIZE).
// ======================================================
#define INPUT_SIZE 64

// ======================================================
// CNN ARCHITECTURE CONSTANTS
// The web trainer page shows the matching #define lines for these values.
// CONV1_FILTERS and CONV2_FILTERS can be changed. The 3x3 kernel is fixed in the loops below,
// so CONV*_KERNEL_SIZE is informational only.
// ======================================================
#define CONV1_KERNEL_SIZE 3
#define CONV1_FILTERS 4
#define CONV1_WEIGHTS (CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE * 3 * CONV1_FILTERS)

#define CONV2_KERNEL_SIZE 3
#define CONV2_FILTERS 8
#define CONV2_WEIGHTS (CONV2_KERNEL_SIZE * CONV2_KERNEL_SIZE * CONV1_FILTERS * CONV2_FILTERS)

// number of weights one conv2 filter owns
#define CONV2_IN_STRIDE (CONV1_FILTERS * 9)

#define CONV1_OUTPUT_SIZE (INPUT_SIZE - 2)
#define POOL1_OUTPUT_SIZE (CONV1_OUTPUT_SIZE / 2)
#define CONV2_OUTPUT_SIZE (POOL1_OUTPUT_SIZE - 2)
#define FLATTENED_SIZE (CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE * CONV2_FILTERS)

#define OUTPUT_WEIGHTS (FLATTENED_SIZE * NUM_CLASSES)

static_assert(INPUT_SIZE % 2 == 0, "INPUT_SIZE must be even");
static_assert(CONV2_OUTPUT_SIZE >= 1, "INPUT_SIZE is too small");

// exact size in bytes of header/myWeights.bin for this sketch's layout
#define MY_EXPECTED_WEIGHT_BYTES ((size_t)(CONV1_WEIGHTS + CONV1_FILTERS + CONV2_WEIGHTS + CONV2_FILTERS + OUTPUT_WEIGHTS + NUM_CLASSES) * 4)

// ======================================================
// GLOBAL VARIABLE DEFINITIONS
// ======================================================

// Add near line 150 with other global buffers:
uint8_t* myRgbBuffer = nullptr;  // Reusable RGB buffer for inference

bool mySDavailable = false;  // set true in setup() if SD mounts ok

// ML Buffers (PSRAM)
float* myInputBuffer = nullptr;
float* myConv1_w = nullptr;
float* myConv1_b = nullptr;
float* myConv2_w = nullptr;
float* myConv2_b = nullptr;
float* myOutput_w = nullptr;
float* myOutput_b = nullptr;

// Gradient buffers
float* myConv1_w_grad = nullptr;
float* myConv1_b_grad = nullptr;
float* myConv2_w_grad = nullptr;
float* myConv2_b_grad = nullptr;
float* myOutput_w_grad = nullptr;
float* myOutput_b_grad = nullptr;

// Adam optimizer momentum buffers
float* myConv1_w_m = nullptr;
float* myConv1_w_v = nullptr;
float* myConv1_b_m = nullptr;
float* myConv1_b_v = nullptr;
float* myConv2_w_m = nullptr;
float* myConv2_w_v = nullptr;
float* myConv2_b_m = nullptr;
float* myConv2_b_v = nullptr;
float* myOutput_w_m = nullptr;
float* myOutput_w_v = nullptr;
float* myOutput_b_m = nullptr;
float* myOutput_b_v = nullptr;

// Forward pass buffers
float* myConv1_output = nullptr;
float* myPool1_output = nullptr;
float* myConv2_output = nullptr;
float* myDense_output = nullptr;

// Backward pass buffers
float* myDense_grad = nullptr;
float* myConv2_grad = nullptr;
float* myPool1_grad = nullptr;
float* myConv1_grad = nullptr;

struct TrainingItem {
  String path;
  int label;
};
std::vector<TrainingItem> myTrainingData;

// ======================================================
// UTILITY FUNCTIONS
// ======================================================
inline float clip_value(float v, float mn=-100, float mx=100) {
  if(isnan(v)||isinf(v)) return 0;
  return constrain(v,mn,mx);
}

inline float leaky_relu(float x) { return x>0 ? x : 0.1f*x; }
inline float leaky_relu_deriv(float x) { return x>0 ? 1.0f : 0.1f; }

// ======================================================
// TOUCH INPUT FUNCTIONS
// ======================================================
int myReadTouch() {
#if MY_TOUCH_ENABLED
  int sum = 0;
  for (int i = 0; i < 3; i++) {
    sum += analogRead(A0);
    delayMicroseconds(100);
  }
  return sum / 3;
#else
  return 0;   // D0 (A0) is the LoRa DIO1 pin, so it is never read as a touch pad
#endif
}

void myResetTouchState() {
  myTouch.isTouching = false;
  myTouch.tapCount = 0;
  myTouch.firstTapTime = 0;
  myTouch.lastReleaseTime = 0;
  myTouch.lastCheckTime = 0;
}

// Background touch monitor that can be called less frequently
void myUpdateTouchState() {
  unsigned long now = millis();
  
  // Only check every 20ms to avoid overwhelming analogRead
  if (now - myTouch.lastCheckTime < 20) return;
  myTouch.lastCheckTime = now;
  
  int val = myReadTouch();
  bool touchActive = myTouch.isTouching 
                      ? (val > myThresholdRelease) 
                      : (val > myThresholdPress);

  // Touch just started
  if (touchActive && !myTouch.isTouching) {
    if (now - myTouch.lastReleaseTime < myTouch.debounceDelay) {
      return; // Debounce
    }
    
    myTouch.isTouching = true;
    
    // First tap or within tap window?
    if (myTouch.tapCount == 0 || (now - myTouch.firstTapTime < myTouch.tapWindow)) {
      if (myTouch.tapCount == 0) {
        myTouch.firstTapTime = now;
      }
      myTouch.tapCount++;
      Serial.printf("Tap #%d\n", myTouch.tapCount);
    } else {
      // Window expired, reset
      myTouch.tapCount = 1;
      myTouch.firstTapTime = now;
      Serial.println("Tap #1 (new window)");
    }
  }
  
  // Touch released
  if (!touchActive && myTouch.isTouching) {
    myTouch.isTouching = false;
    myTouch.lastReleaseTime = now;
  }
}

// Returns: 0=no action, 1=tap, 2=long press (3+ taps)
// NOTE: Always call myUpdateTouchState() before this in tight loops
int myCheckTouchInput() {
  myUpdateTouchState();  // Update state first
  
  unsigned long now = millis();
  
  // Check if tap window expired and we have taps
  if (myTouch.tapCount > 0 && !myTouch.isTouching) {
    if (now - myTouch.firstTapTime > myTouch.tapWindow) {
      int result = (myTouch.tapCount >= myTouch.longPressTaps) ? 2 : 1;
      int count = myTouch.tapCount;
      myResetTouchState();
      
      if (result == 2) {
        Serial.printf("LONG PRESS detected (%d taps)\n", count);
      } else {
        Serial.printf("TAP detected (%d tap%s)\n", count, count > 1 ? "s" : "");
      }
      return result;
    }
  }
  
  return 0;
}

// Non-blocking check - just updates state without consuming events
// Use this in heavy computation loops
void myCheckTouchBackground() {
  myUpdateTouchState();
}

// Check if we have a pending action without consuming it
int myPeekTouchAction() {
  myUpdateTouchState();
  unsigned long now = millis();
  
  if (myTouch.tapCount > 0 && !myTouch.isTouching) {
    if (now - myTouch.firstTapTime > myTouch.tapWindow) {
      return (myTouch.tapCount >= myTouch.longPressTaps) ? 2 : 1;
    }
  }
  return 0;
}




// ======================================================
// MEMORY ALLOCATION
// ======================================================
void myAllocateMemory() {
  if (myInputBuffer != nullptr) return;
  
  Serial.println("\n=== Allocating Memory ===");
  
  myInputBuffer = (float*)ps_malloc(INPUT_SIZE * INPUT_SIZE * 3 * sizeof(float));
  myConv1_w = (float*)ps_malloc(CONV1_WEIGHTS * sizeof(float));
  myConv1_b = (float*)ps_malloc(CONV1_FILTERS * sizeof(float));
  myConv2_w = (float*)ps_malloc(CONV2_WEIGHTS * sizeof(float));
  myConv2_b = (float*)ps_malloc(CONV2_FILTERS * sizeof(float));
  myOutput_w = (float*)ps_malloc(OUTPUT_WEIGHTS * sizeof(float));
  myOutput_b = (float*)ps_malloc(NUM_CLASSES * sizeof(float));

  myConv1_w_grad = (float*)ps_malloc(CONV1_WEIGHTS * sizeof(float));
  myConv1_b_grad = (float*)ps_malloc(CONV1_FILTERS * sizeof(float));
  myConv2_w_grad = (float*)ps_malloc(CONV2_WEIGHTS * sizeof(float));
  myConv2_b_grad = (float*)ps_malloc(CONV2_FILTERS * sizeof(float));
  myOutput_w_grad = (float*)ps_malloc(OUTPUT_WEIGHTS * sizeof(float));
  myOutput_b_grad = (float*)ps_malloc(NUM_CLASSES * sizeof(float));

  myConv1_w_m = (float*)ps_calloc(CONV1_WEIGHTS, sizeof(float));
  myConv1_w_v = (float*)ps_calloc(CONV1_WEIGHTS, sizeof(float));
  myConv1_b_m = (float*)ps_calloc(CONV1_FILTERS, sizeof(float));
  myConv1_b_v = (float*)ps_calloc(CONV1_FILTERS, sizeof(float));
  myConv2_w_m = (float*)ps_calloc(CONV2_WEIGHTS, sizeof(float));
  myConv2_w_v = (float*)ps_calloc(CONV2_WEIGHTS, sizeof(float));
  myConv2_b_m = (float*)ps_calloc(CONV2_FILTERS, sizeof(float));
  myConv2_b_v = (float*)ps_calloc(CONV2_FILTERS, sizeof(float));
  myOutput_w_m = (float*)ps_calloc(OUTPUT_WEIGHTS, sizeof(float));
  myOutput_w_v = (float*)ps_calloc(OUTPUT_WEIGHTS, sizeof(float));
  myOutput_b_m = (float*)ps_calloc(NUM_CLASSES, sizeof(float));
  myOutput_b_v = (float*)ps_calloc(NUM_CLASSES, sizeof(float));

  myConv1_output = (float*)ps_malloc(CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE*CONV1_FILTERS*sizeof(float));
  myPool1_output = (float*)ps_malloc(POOL1_OUTPUT_SIZE*POOL1_OUTPUT_SIZE*CONV1_FILTERS*sizeof(float));
  myConv2_output = (float*)ps_malloc(CONV2_OUTPUT_SIZE*CONV2_OUTPUT_SIZE*CONV2_FILTERS*sizeof(float));
  myDense_output = (float*)ps_malloc(NUM_CLASSES*sizeof(float));

  myDense_grad = (float*)ps_malloc(FLATTENED_SIZE*sizeof(float));
  myConv2_grad = (float*)ps_malloc(CONV2_OUTPUT_SIZE*CONV2_OUTPUT_SIZE*CONV2_FILTERS*sizeof(float));
  myPool1_grad = (float*)ps_malloc(POOL1_OUTPUT_SIZE*POOL1_OUTPUT_SIZE*CONV1_FILTERS*sizeof(float));
  myConv1_grad = (float*)ps_malloc(CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE*CONV1_FILTERS*sizeof(float));

  if (!myInputBuffer || !myConv1_w || !myConv2_w || !myOutput_w || 
      !myConv1_output || !myPool1_output || !myConv2_output) {
    Serial.println("FATAL: PSRAM allocation failed!");
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "PSRAM ERROR!"); } while (u8g2.nextPage());
    while(1) { delay(1000); }
  }

  Serial.printf("Free PSRAM after allocation: %d bytes\n", ESP.getFreePsram());

  // Initialize weights with He initialization
  float c1std = sqrt(2.0/(9.0*3));
  for(int i=0; i<CONV1_WEIGHTS; i++) myConv1_w[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f * c1std;
  for(int i=0; i<CONV1_FILTERS; i++) myConv1_b[i] = 0;
  
  float c2std = sqrt(2.0/(double)CONV2_IN_STRIDE);   // He initialisation
  for(int i=0; i<CONV2_WEIGHTS; i++) myConv2_w[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f * c2std;
  for(int i=0; i<CONV2_FILTERS; i++) myConv2_b[i] = 0;
  
  float dstd = sqrt(2.0/FLATTENED_SIZE);
  for(int i=0; i<OUTPUT_WEIGHTS; i++) myOutput_w[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f * dstd;
  for(int i=0; i<NUM_CLASSES; i++) myOutput_b[i] = 0;
  Serial.println("He-init random weights set");
}

// ======================================================
// WEIGHT SAVE/LOAD
// ======================================================
void myExportHeader() {
  if (!mySDavailable) {
    Serial.println("No SD card - cannot export header");
    return;
  }
  if (!SD.exists("/header")) SD.mkdir("/header");
  File file = SD.open("/header/myWeights.h", FILE_WRITE);
  if (!file) return;
  file.println("#ifndef MY_MODEL_H\n#define MY_MODEL_H");
  file.println("// ======================================================");
  file.println("// IMPORTANT: After copying this file to your sketch folder,");
  file.println("// update ALL of the following lines in your main sketch");
  file.println("// to match the layout, number of classes and labels used during training:");
  file.println("//");
  file.printf( "//   #define INPUT_SIZE %d\n", INPUT_SIZE);
  file.printf( "//   #define CONV1_FILTERS %d\n", CONV1_FILTERS);
  file.printf( "//   #define CONV2_FILTERS %d\n", CONV2_FILTERS);
  file.printf( "//   #define NUM_CLASSES %d\n", NUM_CLASSES);

  file.print("//   String myClassLabels[NUM_CLASSES] = {");
  for (int i = 0; i < NUM_CLASSES; i++) {
    file.printf("\"%s\"", myClassLabels[i].c_str());
    if (i < NUM_CLASSES - 1) file.print(", ");
  }
  file.println("};");

 // file.println("//   String myClassLabels[NUM_CLASSES] = {\"0Blank\", \"1Cup\", \"2Pen\", ...};");
  file.println("//");
  file.println("// Then uncomment:  #define USE_BAKED_WEIGHTS");
  file.println("// ======================================================");
  auto myDump = [&](const char* name, float* data, int size) {
    file.printf("const float %s[] = { ", name);
    for(int i=0; i<size; i++) {
      file.print(data[i], 6); file.print("f");
      if(i < size-1) file.print(", ");
      if((i+1)%8 == 0) file.println();
    }
    file.println(" };");
  };
  myDump("myModel_conv1_w",  myConv1_w,  CONV1_WEIGHTS);
  myDump("myModel_conv1_b",  myConv1_b,  CONV1_FILTERS);
  myDump("myModel_conv2_w",  myConv2_w,  CONV2_WEIGHTS);
  myDump("myModel_conv2_b",  myConv2_b,  CONV2_FILTERS);
  myDump("myModel_output_w", myOutput_w, OUTPUT_WEIGHTS);
  myDump("myModel_output_b", myOutput_b, NUM_CLASSES);
  file.println("#endif");
  Serial.println("You can copy /header/myWeights.h to the sketch folder, then uncomment #define USE_BAKED_WEIGHTS");
  file.close();
}

bool myLoadWeights() {
  if (!mySDavailable) {
    Serial.println("No SD card - skipping weight load");
    return false;
  }
  if (!SD.exists("/header/myWeights.bin")) {
    Serial.println("No SD weights file found");
    return false;
  }
  Serial.println("Loading weights from SD...");
  File f = SD.open("/header/myWeights.bin", FILE_READ);
  if (!f) return false;

  // refuse a weights file that does not match this sketch's layout and class count
  if ((size_t)f.size() != MY_EXPECTED_WEIGHT_BYTES) {
    Serial.printf("REFUSED myWeights.bin: file is %u bytes but this sketch needs %u bytes\n",
                  (unsigned)f.size(), (unsigned)MY_EXPECTED_WEIGHT_BYTES);
    Serial.printf("Sketch layout: INPUT_SIZE %d, CONV1_FILTERS %d, CONV2_FILTERS %d, NUM_CLASSES %d\n",
                  INPUT_SIZE, CONV1_FILTERS, CONV2_FILTERS, NUM_CLASSES);
    Serial.println("Match these #defines to the web trainer page, or retrain. Using random/baked weights instead.");
    f.close();
    return false;
  }

  f.read((uint8_t*)myConv1_w, CONV1_WEIGHTS*4); 
  f.read((uint8_t*)myConv1_b, CONV1_FILTERS*4);
  f.read((uint8_t*)myConv2_w, CONV2_WEIGHTS*4); 
  f.read((uint8_t*)myConv2_b, CONV2_FILTERS*4);
  f.read((uint8_t*)myOutput_w, OUTPUT_WEIGHTS*4); 
  f.read((uint8_t*)myOutput_b, NUM_CLASSES*4);
  f.close();
  Serial.println("Weights loaded successfully");
  myWeightsTrained = true;
  return true;
}

// ======================================================
// READ CLASS LABELS FROM /header/config.json
// The web trainer page writes this file. Only the "classes" list is used, and only when it has exactly
// NUM_CLASSES entries. The sketch's compiled myClassLabels[] are the fallback.
// ======================================================
// ==CFG PARSE START==
static bool myJsonInt(const String& t, const char* key, int& out) {
  String k("\"");
  k += key;
  k += "\"";
  int p = t.indexOf(k.c_str());
  if (p < 0) return false;
  p = t.indexOf(':', p);
  if (p < 0) return false;
  p++;
  int len = (int)t.length();
  while (p < len && (t[p] == ' ' || t[p] == '\n' || t[p] == '\r' || t[p] == '\t')) p++;
  bool neg = false;
  if (p < len && t[p] == '-') { neg = true; p++; }
  if (p >= len || t[p] < '0' || t[p] > '9') return false;
  long v = 0;
  while (p < len && t[p] >= '0' && t[p] <= '9') { v = v * 10 + (t[p] - '0'); p++; }
  out = (int)(neg ? -v : v);
  return true;
}

// Returns the number of strings found in the JSON array named key, or -1 if the key or array is missing.
static int myJsonStringArray(const String& t, const char* key, std::vector<String>& out) {
  String k("\"");
  k += key;
  k += "\"";
  int p = t.indexOf(k.c_str());
  if (p < 0) return -1;
  p = t.indexOf('[', p);
  if (p < 0) return -1;
  p++;
  int len = (int)t.length();
  out.clear();
  while (p < len) {
    char c = t[p];
    if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == ',') { p++; continue; }
    if (c == ']') break;
    if (c != '"') return -1;              // not a list of strings
    p++;
    String s("");
    while (p < len && t[p] != '"') {
      if (t[p] == '\\' && p + 1 < len) p++;   // keep the escaped character
      s += t[p];
      p++;
    }
    if (p >= len) return -1;              // unterminated string
    p++;                                  // closing quote
    out.push_back(s);
  }
  return (int)out.size();
}
// ==CFG PARSE END==

void myLoadConfig() {
  if (!mySDavailable) return;
  if (!SD.exists("/header/config.json")) {
    Serial.println("No /header/config.json - using the class labels compiled into the sketch");
    return;
  }
  File f = SD.open("/header/config.json", FILE_READ);
  if (!f) return;
  size_t sz = f.size();
  if (sz == 0 || sz > 4096) {
    Serial.printf("config.json ignored: unexpected size %u bytes\n", (unsigned)sz);
    f.close();
    return;
  }
  String txt;
  txt.reserve(sz + 1);
  while (f.available()) txt += (char)f.read();
  f.close();

  std::vector<String> names;
  int n = myJsonStringArray(txt, "classes", names);
  if (n < 0) {
    Serial.println("config.json has no readable \"classes\" list - keeping the compiled class labels");
  } else if (n != NUM_CLASSES) {
    Serial.printf("config.json lists %d classes but the sketch has NUM_CLASSES %d - keeping the compiled labels.\n", n, NUM_CLASSES);
    Serial.println("To add or remove classes, change NUM_CLASSES and myClassLabels[] in the sketch and reflash.");
  } else {
    Serial.println("Class labels loaded from /header/config.json:");
    for (int i = 0; i < NUM_CLASSES; i++) {
      myClassLabels[i] = names[i];
      Serial.printf("  %d: %s\n", i, myClassLabels[i].c_str());
    }
  }

  int v;
  if (myJsonInt(txt, "input_size", v) && v != INPUT_SIZE)
    Serial.printf("WARNING: config.json input_size %d but the sketch INPUT_SIZE is %d\n", v, INPUT_SIZE);
  if (myJsonInt(txt, "conv1_filters", v) && v != CONV1_FILTERS)
    Serial.printf("WARNING: config.json conv1_filters %d but the sketch CONV1_FILTERS is %d\n", v, CONV1_FILTERS);
  if (myJsonInt(txt, "conv2_filters", v) && v != CONV2_FILTERS)
    Serial.printf("WARNING: config.json conv2_filters %d but the sketch CONV2_FILTERS is %d\n", v, CONV2_FILTERS);
}

// ======================================================
// WEB SERIAL DEBUG FRAMES
// The web trainer page sends 'D' when it connects and every 5 s, and 'd' when it disconnects.
// While a 'D' was seen in the last 15 s the device prints frame lines that start with "@F":
//   @F <kind> <n> <pred> <probs|-> <logits|-> <layout> <centre RGB|-> <heatSide> <heat base64|0/-> <jpeg base64>
// kind I = inference (every 10th frame), C = image just saved, P = live preview while collecting (about 1/s)
// The JPEG is exactly what the camera produced. Heat = max over conv2 filters, scaled 0..255.
// ======================================================
// ==DBG START==
bool myDebugStream = false;
unsigned long myDebugLastSeen = 0;

static char myCmdBuf[100];
static int myCmdLen = -1;                // -1 = not collecting a command line
static unsigned long myCmdT = 0;

// One hook used by every mode: web page heartbeat (D/d) and @command lines. Returns true when it used the character.
bool myHandleDebugChar(char c) {
  if (myCmdLen >= 0 && millis() - myCmdT > 5000) myCmdLen = -1;   // abandoned line
  if (myCmdLen >= 0) {
    myCmdT = millis();
    if (c == '\n' || c == '\r') {
      myCmdBuf[myCmdLen] = 0;
      myCmdLen = -1;
      myLoraCommand(myCmdBuf);
      if (!myIsSelected) myDrawMenu();   // back to the menu after every @command or >text typed in the menu
    } else if (myCmdLen < (int)sizeof(myCmdBuf) - 1) {
      myCmdBuf[myCmdLen++] = c;
    }
    return true;
  }
  if (c == '@' || c == '>') { myCmdBuf[0] = c; myCmdLen = 1; myCmdT = millis(); return true; }   // '>' starts a LoRa message
  if (c == 'D') {
    if (!myDebugStream) { Serial.println("Debug frames ON"); myLoraPrintInfo(true); }
    myDebugStream = true;
    myDebugLastSeen = millis();
    return true;
  }
  if (c == 'd') {
    if (myDebugStream) Serial.println("Debug frames OFF");
    myDebugStream = false;
    return true;
  }
  return false;
}

static void myPrintB64(const uint8_t* d, size_t n) {
  unsigned char out[520];                  // 384 input bytes -> 512 characters + terminator
  while (n > 0) {
    size_t take = n > 384 ? 384 : n;       // multiple of 3, so the chunks join into one valid base64 string
    size_t ol = 0;
    mbedtls_base64_encode(out, sizeof(out), &ol, d, take);
    Serial.write(out, ol);
    d += take;
    n -= take;
  }
}

void myDebugSendFrame(char kind, int n, camera_fb_t* fb, int pred, const float* logits) {
  if (!myDebugStream || !fb || !Serial) return;
  if (millis() - myDebugLastSeen > 15000) { myDebugStream = false; return; }   // page stopped sending heartbeats
  const bool inf = (kind == 'I' && logits != nullptr);
  Serial.printf("@F %c %d %d ", kind, n, pred);
  if (inf) {
    for (int i = 0; i < NUM_CLASSES; i++) { if (i) Serial.print(','); Serial.print(myDense_output[i], 4); }
    Serial.print(' ');
    for (int i = 0; i < NUM_CLASSES; i++) { if (i) Serial.print(','); Serial.print(logits[i], 4); }
  } else {
    Serial.print("- -");
  }
  Serial.printf(" %dx%dx%d ", INPUT_SIZE, CONV1_FILTERS, CONV2_FILTERS);
  if (inf) {
    int c = ((INPUT_SIZE / 2) * INPUT_SIZE + INPUT_SIZE / 2) * 3;    // centre pixel of the model input
    Serial.print(myInputBuffer[c], 4); Serial.print(',');
    Serial.print(myInputBuffer[c + 1], 4); Serial.print(',');
    Serial.print(myInputBuffer[c + 2], 4);
  } else {
    Serial.print('-');
  }
  Serial.print(' ');
  if (inf) {
    const int hn = CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE;
    static uint8_t heat[CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE];
    float lo = 1e30f, hi = -1e30f;
    for (int i = 0; i < hn; i++) {
      float m = myConv2_output[i];
      for (int f = 1; f < CONV2_FILTERS; f++) { float v = myConv2_output[f * hn + i]; if (v > m) m = v; }
      if (m < lo) lo = m;
      if (m > hi) hi = m;
    }
    float span = hi - lo;
    if (span < 1e-9f) span = 1.0f;
    for (int i = 0; i < hn; i++) {
      float m = myConv2_output[i];
      for (int f = 1; f < CONV2_FILTERS; f++) { float v = myConv2_output[f * hn + i]; if (v > m) m = v; }
      heat[i] = (uint8_t)(255.0f * (m - lo) / span + 0.5f);
    }
    Serial.printf("%d ", CONV2_OUTPUT_SIZE);
    myPrintB64(heat, hn);
  } else {
    Serial.print("0 -");
  }
  Serial.print(' ');
  myPrintB64(fb->buf, fb->len);
  Serial.println();
}
// ==DBG END==

// ==LORA START==
// ======================================================
// LORA: summaries, messages and commands
// SX1262 settings: 915 MHz + channel x 0.1 MHz, SF9, BW 125 kHz, CR 4/7, 22 dBm, sync word 0x3444.
// Pins and global variables are declared in the LORA DECLARATIONS block near the top of the sketch.
// ======================================================
void IRAM_ATTR myLoraIsr() { myLoraFlag = true; }

float myLoraFreq() { return 915.0f + myLoraChannel * 0.1f; }

// Printable-ASCII Vigenere cipher: hides text from casual listeners, it is NOT strong security.
void myLoraCipher(char* b, bool enc) {
  int sl = strlen(myLoraSeed);
  if (sl == 0) return;
  for (int i = 0; b[i]; i++) {
    char c = b[i];
    if (c >= 32 && c <= 126) {
      int sh = myLoraSeed[i % sl] % 95;
      if (!enc) sh = -sh;
      b[i] = 32 + (c - 32 + sh + 95) % 95;
    }
  }
}

void myLoraSave() {
  myPrefs.begin("lora", false);
  myPrefs.putString("name", myLoraName);
  myPrefs.putInt("ch", myLoraChannel);
  myPrefs.putBool("enc", myLoraEncrypt);
  myPrefs.putString("seed", myLoraSeed);
  myPrefs.putInt("rep", myLoraReportSec);
  myPrefs.putInt("conf", myLoraMinConf);
  myPrefs.putBool("auto", myAutoStart);
  myPrefs.putInt("adly", myAutoDelay);
  myPrefs.putInt("prt", myPrintEvery);
  myPrefs.putBool("tch", myTouchExit);
  myPrefs.end();
}

void myLoraLoad() {
  myPrefs.begin("lora", false);
  myPrefs.getString("name", MY_DEFAULT_NAME).toCharArray(myLoraName, sizeof(myLoraName));
  myLoraChannel   = myPrefs.getInt("ch", MY_DEFAULT_CHANNEL);
  myLoraEncrypt   = myPrefs.getBool("enc", MY_DEFAULT_ENCRYPT);
  myPrefs.getString("seed", MY_DEFAULT_SEED).toCharArray(myLoraSeed, sizeof(myLoraSeed));
  myLoraReportSec = myPrefs.getInt("rep", MY_DEFAULT_REPORT_SEC);
  myLoraMinConf   = myPrefs.getInt("conf", MY_DEFAULT_MIN_CONF);
  myAutoStart     = myPrefs.getBool("auto", MY_DEFAULT_AUTO_START);
  myAutoDelay     = myPrefs.getInt("adly", MY_DEFAULT_AUTO_DELAY_S);
  myPrintEvery    = myPrefs.getInt("prt", MY_DEFAULT_PRINT_EVERY);
  myTouchExit     = myPrefs.getBool("tch", MY_DEFAULT_TOUCH_EXIT);
  myPrefs.end();
}

void myLoraPrintInfo(bool force) {
  if (!force && !myDebugStream) return;   // machine line, only for the web page
  myLoraInfoLast = millis();
  Serial.printf("@LORA-INFO name=%s ch=%d report=%d conf=%d radio=%s auto=%d classes=", myLoraName, myLoraChannel,
                myLoraReportSec, myLoraMinConf, myLoraOk ? "ok" : "off", myAutoStart ? 1 : 0);
  for (int i = 0; i < NUM_CLASSES; i++) { if (i) Serial.print(','); Serial.print(myClassLabels[i]); }
  Serial.println();
}

void myLoraResetDefaults() {
  myPrefs.begin("lora", false); myPrefs.clear(); myPrefs.end();
  myLoraLoad();
  if (myLoraOk) { myRadio.standby(); myRadio.setFrequency(myLoraFreq()); myRadio.startReceive(); }
  myLoraNext = millis() + myLoraReportSec * 1000UL;
  Serial.println(F("[OK] all settings are back to the defaults"));
  myLoraHelp(); myLoraPrintInfo(false);
}

void myLoraBegin() {
  myLoraLoad();
  Serial.print(F("[SX1262] init ... "));
  int st = myRadio.begin(myLoraFreq(), 125.0, 9, 7, RADIOLIB_SX126X_SYNC_WORD_PRIVATE, 22, 8, MY_LORA_TCXO_V);
  if (st != RADIOLIB_ERR_NONE) {
    Serial.printf("failed, code %d. LoRa is OFF (check wiring). Summaries are still printed for the web page.\n", st);
  } else {
    myRadio.setDio2AsRfSwitch(MY_LORA_DIO2_RF);
#if MY_LORA_RXEN >= 0 || MY_LORA_TXEN >= 0
    myRadio.setRfSwitchPins(MY_LORA_RXEN, MY_LORA_TXEN);
#endif
    myRadio.setSyncWord(0x3444);
    myRadio.setDio1Action(myLoraIsr);
    myLoraOk = (myRadio.startReceive() == RADIOLIB_ERR_NONE);
    Serial.println(myLoraOk ? F("ready") : F("receive failed"));
  }
  myLoraWinStart = millis();
  myLoraNext = millis() + 5000 + random(myLoraReportSec * 1000L);   // random first report so devices do not collide
  myLoraPrintInfo(false);
}

// Called from the inference loop for every frame.
void myLoraCount(int pred, float p) {
  myLoraFrames++;
  if (pred >= 0 && pred < NUM_CLASSES && p * 100.0f >= myLoraMinConf) myLoraCounts[pred]++;
}

// ---- time stamps, readable packet descriptions, last messages for the OLED ----
long myLoraClockBase = -1;            // -1 = no clock set: stamps show time since boot; else seconds-of-day minus uptime
char myLoraLastTx[40] = "", myLoraLastRx[40] = "";   // "<stamp> <text>" of the last message, for the OLED

void myLoraStamp(char* out, size_t n) {
  unsigned long s = millis() / 1000UL;
  if (myLoraClockBase >= 0) {
    s = (s + (unsigned long)myLoraClockBase) % 86400UL;
    snprintf(out, n, "%02lu:%02lu:%02lu", s / 3600, (s / 60) % 60, s % 60);
  } else {
    snprintf(out, n, "T+%02lu:%02lu:%02lu", s / 3600, (s / 60) % 60, s % 60);
  }
}

// Turns a summary packet  S,<name>,<seq>,<period>,<frames>,<c0>,<c1>,...  into words, e.g.
//   summary #5: 30 s window, 215 frames -> 0Blank 67, 1Cup 120, 2Pen 0, unsure 28
// The sender name goes into 'name'. Returns false when the packet is not a summary.
bool myLoraExplain(const char* pkt, char* name, size_t nn, char* out, size_t n) {
  if (pkt[0] != 'S' || pkt[1] != ',') return false;
  char tmp[MY_LORA_MAX + 40];
  strncpy(tmp, pkt, sizeof(tmp) - 1); tmp[sizeof(tmp) - 1] = 0;
  char* sp;
  strtok_r(tmp, ",", &sp);                              // "S"
  char* nm  = strtok_r(NULL, ",", &sp);
  char* seq = strtok_r(NULL, ",", &sp);
  char* per = strtok_r(NULL, ",", &sp);
  char* frm = strtok_r(NULL, ",", &sp);
  if (!nm || !seq || !per || !frm) return false;
  strncpy(name, nm, nn - 1); name[nn - 1] = 0;
  long frames = atol(frm), sum = 0;
  int len = snprintf(out, n, "summary #%s: %s s window, %s frames ->", seq, per, frm);
  int i = 0;
  for (char* c = strtok_r(NULL, ",", &sp); c && len < (int)n - 24; c = strtok_r(NULL, ",", &sp), i++) {
    long v = atol(c);
    sum += v;
    if (i < NUM_CLASSES) len += snprintf(out + len, n - len, "%s %s %ld", i ? "," : "", myClassLabels[i].c_str(), v);
    else                 len += snprintf(out + len, n - len, "%s class%d %ld", i ? "," : "", i, v);
  }
  if (frames > sum && len < (int)n - 24) snprintf(out + len, n - len, ", unsure %ld", frames - sum);
  return true;
}

// Sends one packet and prints what was sent and when. Returns true when the radio accepted it.
bool myLoraTransmit(char* pkt) {
  char st[16]; myLoraStamp(st, sizeof(st));
  int len = strlen(pkt);
  if (!myLoraOk)    { Serial.printf("[%s] LoRa NOT SENT, the radio is off (check wiring): %s\n", st, pkt); return false; }
  if (myLoraTxBusy) { Serial.printf("[%s] LoRa NOT SENT, the radio is still busy, try again: %s\n", st, pkt); return false; }
  unsigned long air = (unsigned long)(myRadio.getTimeOnAir(len) / 1000);
  char nm[24], ex[200];
  const char* enc = myLoraEncrypt ? ", encrypted" : "";
  bool isSummary = myLoraExplain(pkt, nm, sizeof(nm), ex, sizeof(ex));
  if (isSummary) Serial.printf("\n[%s] LoRa SENT %s  (%d bytes, ~%lu ms on air%s)\n", st, ex, len, air, enc);
  else           Serial.printf("\n[%s] LoRa SENT message: \"%s\"  (%d bytes, ~%lu ms on air%s)\n", st, pkt, len, air, enc);
  char plain[MY_LORA_MAX + 40];
  strncpy(plain, pkt, sizeof(plain) - 1); plain[sizeof(plain) - 1] = 0;
  snprintf(myLoraLastTx, sizeof(myLoraLastTx), "%s %s", st, pkt);
  if (myLoraEncrypt) myLoraCipher(pkt, true);
  digitalWrite(MY_SD_CS, HIGH);   // SD card deselected before LoRa SPI activity
  myRadio.standby();
  if (myRadio.startTransmit(pkt) == RADIOLIB_ERR_NONE) {
    myLoraTxBusy = true; myLoraTxStart = millis(); myLoraTxN++;
    if (!isSummary) Serial.printf("@LORA-MSG self 0 %s\n", plain);   // machine line for the web page
    return true;
  }
  myLoraErrN++; myRadio.startReceive();
  Serial.printf("[%s] LoRa SEND FAILED, the radio did not start sending\n", st);
  return false;
}

// Chat line: "<name>: <text>"
bool myLoraSendText(const char* text) {
  while (*text == ' ') text++;
  if (!*text) { Serial.println(F("[E] nothing to send. Type some text after the command")); return false; }
  char pkt[MY_LORA_MAX + 1];
  int need = snprintf(pkt, sizeof(pkt), "%s: %s", myLoraName, text);
  if (need >= (int)sizeof(pkt)) Serial.printf("[note] message cut to %d characters (the limit includes \"%s: \")\n", (int)sizeof(pkt) - 1, myLoraName);
  return myLoraTransmit(pkt);
}

void myLoraReport() {
  unsigned long now = millis();
  unsigned long period = (now - myLoraWinStart + 500) / 1000;
  myLoraNext = now + (unsigned long)myLoraReportSec * 10UL * (90 + random(21));   // 90..110 % of the period
  if (myLoraFrames == 0) { myLoraWinStart = now; return; }                       // not inferring: nothing to say
  char pkt[MY_LORA_MAX];
  int n = snprintf(pkt, sizeof(pkt), "S,%s,%lu,%lu,%lu", myLoraName, myLoraSeq, period, (unsigned long)myLoraFrames);
  for (int i = 0; i < NUM_CLASSES && n < (int)sizeof(pkt) - 8; i++) n += snprintf(pkt + n, sizeof(pkt) - n, ",%u", myLoraCounts[i]);
  Serial.printf("@LORA self 0 %s\n", pkt);
  if (myLoraOk) myLoraTransmit(pkt);   // prints a time-stamped explanation of what was sent
  myLoraSeq++;
  memset(myLoraCounts, 0, sizeof(myLoraCounts));
  myLoraFrames = 0;
  myLoraWinStart = now;
}

void myLoraService() {
  unsigned long now = millis();
  if (myLoraOk && myLoraFlag) {
    myLoraFlag = false;
    if (myLoraTxBusy) {
      myLoraTxBusy = false;
      myRadio.startReceive();
    } else {
      String s;
      int st = myRadio.readData(s);
      if (st == RADIOLIB_ERR_NONE) {
        myLoraRxN++;
        char b[MY_LORA_MAX + 40];
        s.toCharArray(b, sizeof(b));
        if (myLoraEncrypt) myLoraCipher(b, false);
        int rssi = (int)myRadio.getRSSI(); float snr = myRadio.getSNR();
        char stp[16], nm[24], ex[200];
        myLoraStamp(stp, sizeof(stp));
        if (b[0] == 'S' && b[1] == ',') {
          Serial.printf("@LORA %d %.1f %s\n", rssi, snr, b);   // machine line for the web page (unchanged)
          if (myLoraExplain(b, nm, sizeof(nm), ex, sizeof(ex))) Serial.printf("\n[%s] LoRa HEARD %s: %s  (%d dBm, SNR %.1f)\n", stp, nm, ex, rssi, snr);
        } else {
          Serial.printf("\n[%s] LoRa HEARD message: \"%s\"  (%d dBm, SNR %.1f)\n", stp, b, rssi, snr);
          Serial.printf("@LORA-MSG %d %.1f %s\n", rssi, snr, b);   // machine line for the web page
          snprintf(myLoraLastRx, sizeof(myLoraLastRx), "%s %s", stp, b);
        }
      } else if (st != RADIOLIB_ERR_RX_TIMEOUT) {
        myLoraErrN++;
        Serial.printf("[E] LoRa read error %d\n", st);
      }
      myRadio.startReceive();
    }
  }
  if (myLoraTxBusy && now - myLoraTxStart > 3000) { myLoraTxBusy = false; myLoraErrN++; myRadio.startReceive(); }
  if (!myLoraTxBusy && (long)(now - myLoraNext) >= 0) myLoraReport();
  if (myDebugStream && now - myLoraInfoLast > 30000) myLoraPrintInfo(false);
}

// Commands together with the current value of every setting ([saved] = changed from the default and kept in flash)
void myLoraHelp() {
  myPrefs.begin("lora", false);
  char v[48];
  auto row = [&](const char* cmd, const char* what, const char* val, const char* key) {
    Serial.printf("  %-18s %-38s now: %s  [%s]\n", cmd, what, val, myPrefs.isKey(key) ? "saved" : "default");
  };
  Serial.println(F("\n=== COMMANDS (type, then Enter) ==="));
  Serial.println(F("SEND A MESSAGE"));
  Serial.println(F("  >your text         send a LoRa message to every board that is listening (works in any mode)"));
  Serial.println(F("SETTINGS (each one is saved in flash; @reset restores the defaults)"));
  row("@name <text>", "this board's name in every report", myLoraName, "name");
  snprintf(v, sizeof(v), "%d s", myLoraReportSec);                       row("@report <sec>", "seconds between LoRa reports (5-3600)", v, "rep");
  snprintf(v, sizeof(v), "%d = %.1f MHz", myLoraChannel, myLoraFreq());  row("@channel <0-120>", "LoRa channel, all boards must match", v, "ch");
  snprintf(v, sizeof(v), "%d %%", myLoraMinConf);                        row("@conf <pct>", "confidence needed to count a frame", v, "conf");
  row("@autostart on|off", "inference starts by itself after power-up", myAutoStart ? "on" : "off", "auto");
  snprintf(v, sizeof(v), "%d s", myAutoDelay);                           row("@autodelay <sec>", "countdown before that auto-start (0-120)", v, "adly");
  snprintf(v, sizeof(v), "every %d. frame", myPrintEvery);               row("@print <n>", "print every Nth frame while inferring", v, "prt");
#if MY_TOUCH_ENABLED
  row("@touch on|off", "touch pad A0 can leave inference", myTouchExit ? "on" : "off", "tch");
#else
  row("@touch on|off", "touch pad A0 can leave inference", "n/a, touch pad off (D0 = LoRa DIO1)", "tch");
#endif
  row("@encrypt on|off", "scramble LoRa text with the seed", myLoraEncrypt ? "on" : "off", "enc");
  row("@seed <text>", "encryption seed, same on all boards", myLoraSeed, "seed");
  Serial.println(F("OTHER"));
  Serial.println(F("  @menu              show the main menu again (also leaves the LoRa msg screen)"));
  Serial.println(F("  @status            radio state, packet counters and the time"));
  Serial.println(F("  @time hh:mm[:ss]   set the clock used in the [time] stamps (lost at reboot, default is time since boot)"));
  Serial.println(F("  @reset             every setting back to the defaults (defaults: USER SETTINGS at the top of the sketch)"));
  Serial.println(F("  @help              this list"));
  myPrefs.end();
}

void myLoraStatus() {
  char st[16]; myLoraStamp(st, sizeof(st));
  Serial.println(F("\n=== STATUS ==="));
  Serial.printf("  board %s   radio %s   channel %d (%.1f MHz)   encryption %s\n", myLoraName, myLoraOk ? "ok" : "OFF (check wiring)", myLoraChannel, myLoraFreq(), myLoraEncrypt ? "on" : "off");
  Serial.printf("  LoRa packets: sent %lu, heard %lu, errors %lu     summaries sent: %lu (one every %d s)\n", myLoraTxN, myLoraRxN, myLoraErrN, myLoraSeq, myLoraReportSec);
  Serial.printf("  time now %s%s\n", st, myLoraClockBase >= 0 ? "" : "   (time since boot; @time hh:mm sets the clock)");
  Serial.printf("  OLED %s\n", u8g2.present ? "ok" : "not found, screen output is skipped");
}

void myLoraCommand(char* l) {
  char chbuf[16];
  if (!strncasecmp(l, "@channel", 8)) { const char* a = l + 8; while (*a == ' ') a++; snprintf(chbuf, sizeof(chbuf), "@%s", a); l = chbuf; }   // @channel 3 = @3
  if (!strcasecmp(l, "@help") || !strcmp(l, "@?") || !strcasecmp(l, "@settings")) { myLoraHelp(); return; }
  if (!strcasecmp(l, "@menu")) { myWantMenu = true; return; }                       // the menu loop reprints itself
  if (!strcasecmp(l, "@status") || !strcasecmp(l, "@info") || !strcasecmp(l, "@stats")) { myLoraStatus(); return; }
  if (!strcasecmp(l, "@pageinfo")) { myLoraPrintInfo(true); return; }               // machine line, used by the web page
  if (!strcasecmp(l, "@reset")) { myLoraResetDefaults(); return; }
  if (!strncasecmp(l, "@autodelay", 10)) {
    const char* a = l + 10; while (*a == ' ') a++;
    int v = atoi(a);
    if (!*a || v < 0 || v > 120) { Serial.println(F("[E] use  @autodelay <seconds>  (0..120)")); return; }
    myAutoDelay = v; myLoraSave(); Serial.printf("[OK] auto-start countdown %d s (saved)\n", v); myLoraPrintInfo(false); return;
  }
  if (!strncasecmp(l, "@print", 6)) {
    const char* a = l + 6; while (*a == ' ') a++;
    int v = atoi(a);
    if (!*a || v < 1 || v > 1000) { Serial.println(F("[E] use  @print <n>  (1 = every frame, 10 = every 10th frame)")); return; }
    myPrintEvery = v; myLoraSave(); Serial.printf("[OK] while inferring, print every %d. frame (saved)\n", v); myLoraPrintInfo(false); return;
  }
  if (!strncasecmp(l, "@touch", 6)) {
    const char* a = l + 6; while (*a == ' ') a++;
    if (!strcasecmp(a, "on") || !strcmp(a, "1")) myTouchExit = true;
    else if (!strcasecmp(a, "off") || !strcmp(a, "0")) myTouchExit = false;
    else { Serial.println(F("[E] use  @touch on  or  @touch off")); return; }
    myLoraSave(); Serial.printf("[OK] touch pad can leave inference: %s (saved)\n", myTouchExit ? "on" : "off"); myLoraPrintInfo(false); return;
  }
  if (!strncasecmp(l, "@autostart", 10)) {
    const char* a = l + 10; while (*a == ' ') a++;
    if (!strcasecmp(a, "on") || !strcmp(a, "1")) myAutoStart = true;
    else if (!strcasecmp(a, "off") || !strcmp(a, "0")) myAutoStart = false;
    else { Serial.println(F("[E] use  @autostart on  or  @autostart off")); return; }
    myLoraSave();
    Serial.printf("[OK] auto-start inference after power-up: %s (saved)\n", myAutoStart ? "ON" : "off");
    myLoraPrintInfo(false); return;
  }
  if (!strncasecmp(l, "@name ", 6)) {
    char* p = l + 6; while (*p == ' ') p++;
    int n = 0;
    for (; p[n] && n < (int)sizeof(myLoraName) - 1; n++) {
      char c = p[n];
      myLoraName[n] = (isalnum((unsigned char)c) || c == '-' || c == '_') ? c : '-';
    }
    myLoraName[n] = 0;
    if (n == 0) strcpy(myLoraName, MY_DEFAULT_NAME);
    myLoraSave(); Serial.printf("[OK] name = %s (saved)\n", myLoraName); myLoraPrintInfo(false); return;
  }
  if (!strncasecmp(l, "@report ", 8)) {
    int v = atoi(l + 8);
    if (v < 5 || v > 3600) { Serial.println(F("[E] report must be 5..3600 seconds")); return; }
    myLoraReportSec = v; myLoraSave(); myLoraNext = millis() + v * 1000UL;
    Serial.printf("[OK] report every %d s (saved)\n", v); myLoraPrintInfo(false); return;
  }
  if (!strncasecmp(l, "@conf ", 6)) {
    int v = atoi(l + 6);
    if (v < 0 || v > 100) { Serial.println(F("[E] conf must be 0..100")); return; }
    myLoraMinConf = v; myLoraSave(); Serial.printf("[OK] min confidence %d%% (saved)\n", v); myLoraPrintInfo(false); return;
  }
  if (!strncasecmp(l, "@seed ", 6)) {
    strncpy(myLoraSeed, l + 6, sizeof(myLoraSeed) - 1); myLoraSeed[sizeof(myLoraSeed) - 1] = 0;
    myLoraSave(); Serial.println(F("[OK] seed updated (saved)")); return;
  }
  if (!strcasecmp(l, "@encrypt on") || !strcasecmp(l, "@encrypt off")) {
    myLoraEncrypt = (l[10] == 'n' || l[10] == 'N'); myLoraSave();
    Serial.println(myLoraEncrypt ? F("[OK] encryption ON (saved)") : F("[OK] encryption OFF (saved)")); return;
  }
  if (l[0] == '>') { myLoraSendText(l + 1); return; }                        // >hello
  if (!strncasecmp(l, "@say ", 5)) { myLoraSendText(l + 5); return; }
  if (!strncasecmp(l, "@time", 5)) {
    int h, m, sec = 0;
    if (l[5] == 0) { char st[16]; myLoraStamp(st, sizeof(st)); Serial.printf("[%s] now\n", st); return; }
    int got = sscanf(l + 5, "%d:%d:%d", &h, &m, &sec);
    if (got < 2 || h < 0 || h > 23 || m < 0 || m > 59 || sec < 0 || sec > 59) { Serial.println(F("[E] use  @time hh:mm  or  @time hh:mm:ss  (24 hour clock)")); return; }
    long tod = h * 3600L + m * 60L + sec;
    myLoraClockBase = (tod + 86400L - (long)((millis() / 1000UL) % 86400UL)) % 86400L;
    Serial.println(F("[OK] clock set, [time] stamps now show the time of day (not saved, lost at reboot)")); return;
  }
  char* e;
  long ch = strtol(l + 1, &e, 10);
  if (*e == 0 && e != l + 1 && ch >= 0 && ch <= 120) {
    myLoraChannel = (int)ch; myLoraSave();
    if (myLoraOk) { myRadio.standby(); int st = myRadio.setFrequency(myLoraFreq()); myRadio.startReceive();
      Serial.printf(st == RADIOLIB_ERR_NONE ? "[OK] channel %d (%.1f MHz, saved)\n" : "[E] frequency change failed (%d)\n", st == RADIOLIB_ERR_NONE ? (int)ch : st, myLoraFreq()); }
    myLoraPrintInfo(false); return;
  }
  Serial.println(F("[E] Unknown command. Type @help"));
}
// ==LORA END==


void mySaveWeights() {
  if (!mySDavailable) {
    Serial.println("No SD card - cannot save weights");
    return;
  }
  if (!SD.exists("/header")) SD.mkdir("/header");
  File f = SD.open("/header/myWeights.bin", FILE_WRITE);
  if (f) {
    f.write((uint8_t*)myConv1_w, CONV1_WEIGHTS*4); 
    f.write((uint8_t*)myConv1_b, CONV1_FILTERS*4);
    f.write((uint8_t*)myConv2_w, CONV2_WEIGHTS*4); 
    f.write((uint8_t*)myConv2_b, CONV2_FILTERS*4);
    f.write((uint8_t*)myOutput_w, OUTPUT_WEIGHTS*4); 
    f.write((uint8_t*)myOutput_b, NUM_CLASSES*4);
    f.close();
    Serial.println("Weights saved to SD");
  }
  myExportHeader();
}

// ======================================================
// IMAGE LOADING FROM SD
// ======================================================
bool myLoadImageFromFile(const char* path, float* buf) {
  File f = SD.open(path);
  if(!f) return false;
  
  size_t sz = f.size();
  uint8_t* jpg = (uint8_t*)ps_malloc(sz);
  if(!jpg) { f.close(); return false; }
  f.read(jpg, sz);
  f.close();
  
  // Use the pre-allocated global myRgbBuffer: allocating 172 KB of PSRAM on every image load
  // is slow, fragments PSRAM and makes touch/serial feel unresponsive.
  if(!myRgbBuffer) { free(jpg); return false; }
  
  bool ok = fmt2rgb888(jpg, sz, PIXFORMAT_JPEG, myRgbBuffer);
  free(jpg);
  if(!ok) return false;
  
  for(int y=0; y<INPUT_SIZE; y++) {
    for(int x=0; x<INPUT_SIZE; x++) {
      int sy = (int)((y+0.5)*240.0/INPUT_SIZE);
      int sx = (int)((x+0.5)*240.0/INPUT_SIZE);
      if(sy>239) sy=239;
      if(sx>239) sx=239;
      int srcIdx = (sy*240 + sx)*3;
      int dstIdx = (y*INPUT_SIZE + x)*3;
      buf[dstIdx]   = myRgbBuffer[srcIdx]   / 255.0f;
      buf[dstIdx+1] = myRgbBuffer[srcIdx+1] / 255.0f;
      buf[dstIdx+2] = myRgbBuffer[srcIdx+2] / 255.0f;
    }
  }
  return true;
}

// ======================================================
// PART 0: SETUP AND LOOP
// ======================================================

// Forward declarations for functions defined in other parts
void myActionCollect(int classIdx);
void myActionTrain();
void myActionInfer();
void myResetMenuState();
void myHandleMenuNavigation();
void myDrawMenu();

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000); 
  delay(1000);  // slow down the startup
  
  Serial.println("\n=== XIAO ESP32-S3 ML System Starting (firmware-lora-v007) ===");
  Serial.printf("Layout: INPUT_SIZE %d, CONV1_FILTERS %d, CONV2_FILTERS %d, NUM_CLASSES %d\n",
                INPUT_SIZE, CONV1_FILTERS, CONV2_FILTERS, NUM_CLASSES);
  Serial.printf("Free heap: %d bytes\n", ESP.getFreeHeap());
  Serial.printf("Free PSRAM: %d bytes\n", ESP.getFreePsram());
  
// Add in setup() function:
myRgbBuffer = (uint8_t*)ps_malloc(240*240*3);
if (!myRgbBuffer) {
  Serial.println("Failed to allocate RGB buffer!");
}

#if MY_TOUCH_ENABLED
  pinMode(A0, INPUT);
#endif
  if (myOledProbe()) {
    u8g2.begin();
    Serial.println("OLED found");
  } else {
    u8g2.present = false;
    Serial.println("OLED not found on D4/D5, screen output is skipped. The Serial Monitor and the web page still work.");
  }
  
// Manual SPI init with timeout to prevent hang when no SD card present
  pinMode(MY_SD_CS, OUTPUT);
  digitalWrite(MY_SD_CS, HIGH);
  delay(100);

  pinMode(LORA_NSS, OUTPUT); digitalWrite(LORA_NSS, HIGH);   // LoRa chip stays quiet while the SD card starts
  Serial.println("Checking SD card...");
  // shared SPI bus (SD card + LoRa) with explicit pins and NO hardware SS
  SPI.begin(LORA_SCK, LORA_MISO, LORA_MOSI, -1);
  
  mySDavailable = SD.begin(MY_SD_CS, SPI, 400000, "/sd", 5, false);
  
  if (!mySDavailable) {
      SD.end();   // instead of SPI.end()
    Serial.println("No SD card - continuing without it");
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "No SD card"); } while (u8g2.nextPage());
    delay(2000);
  } else {
    Serial.println("SD card mounted successfully");
    myLoadConfig();   // class labels from /header/config.json
  }

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM; config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM; config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM; config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM; config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM; config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM; config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM; config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM; config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000; config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_240X240; config.jpeg_quality = 12;
  config.fb_count = 1;     // 2
  esp_err_t camErr = esp_camera_init(&config);
  if (camErr != ESP_OK) {
    Serial.printf("Camera init FAILED: 0x%x\n", camErr);
  } else {
    Serial.println("Camera initialized");
  }
  sensor_t * s = esp_camera_sensor_get();
    if (s != NULL) {
      // mirrored + flipped vertically to match the web trainer page, and brighter
      s->set_hmirror(s, MY_CAM_HMIRROR);
      s->set_vflip(s, MY_CAM_VFLIP);
      s->set_brightness(s, MY_CAM_BRIGHTNESS);   // -2..2
      s->set_ae_level(s, MY_CAM_AE_LEVEL);       // -2..2
    }

  // throw away the first few frames so auto exposure settles before any image is used
  for (int i = 0; i < MY_CAM_WARMUP_FRAMES; i++) {
    camera_fb_t* warm = esp_camera_fb_get();
    if (warm) esp_camera_fb_return(warm);
    delay(60);
  }

  // ESP-IDF Log Levels (ordered least to most verbose):
  //   ESP_LOG_NONE    (0) — no output at all
  //   ESP_LOG_WARN    (2) — errors + W(...) warnings
  //   ESP_LOG_VERBOSE (5) — + V(...) everything

  // Set globally first, then override specific tags as needed:
  esp_log_level_set("*", ESP_LOG_WARN);           // suppress INFO spam globally
  esp_log_level_set("esp_camera", ESP_LOG_ERROR); // suppress FB_OVF (WARN level)

  myAllocateMemory();  // allocates PSRAM and sets random He-init weights

  // SD card and camera init can leave the SPI pins changed, so set the shared bus up again and keep
  // both chip selects HIGH before the radio starts
  SPI.begin(LORA_SCK, LORA_MISO, LORA_MOSI, -1);
  digitalWrite(MY_SD_CS, HIGH);
  digitalWrite(LORA_NSS, HIGH);
  myLoraBegin();       // start the LoRa radio

#ifdef USE_BAKED_WEIGHTS
  memcpy(myConv1_w,  myModel_conv1_w,  CONV1_WEIGHTS  * sizeof(float));
  memcpy(myConv1_b,  myModel_conv1_b,  CONV1_FILTERS  * sizeof(float));
  memcpy(myConv2_w,  myModel_conv2_w,  CONV2_WEIGHTS  * sizeof(float));
  memcpy(myConv2_b,  myModel_conv2_b,  CONV2_FILTERS  * sizeof(float));
  memcpy(myOutput_w, myModel_output_w, OUTPUT_WEIGHTS * sizeof(float));
  memcpy(myOutput_b, myModel_output_b, NUM_CLASSES    * sizeof(float));
  Serial.println("Baked-in weights loaded from myModel.h");
  myWeightsTrained = true; 
#endif

  if (myLoadWeights()) {
    Serial.println("SD weights loaded - overriding baked-in weights");
  }


  myLastActivityTime = millis();
  myIsSelected = false; myResetTouchState();   // the menu is printed once, below
  delay(2000);  // time to get things started like the serial monitor

#if MY_TOUCH_ENABLED
  Serial.println("System ready - tap A0 to navigate, 3+ taps to select, or use t / l / digits in the Serial Monitor");
#else
  Serial.println("System ready - use t (next), l (select) or a digit in the Serial Monitor");
#endif
  myLoraHelp();            // commands with their current values first, the menu last so it sits next to the cursor
  myDrawMenu();

}

void loop() {
  static bool myBootAuto = true;
  myLoraService();
  if (myBootAuto) {
    myBootAuto = false;
    if (myAutoStart && myWeightsTrained) {
      Serial.printf("\nInference starts by itself in %d s. Send t, l or a digit to stay in the menu. (@autostart off turns this off)\n", myAutoDelay);
      unsigned long t0 = millis(); bool cancel = false;
      while (!cancel && millis() - t0 < myAutoDelay * 1000UL) {
        myLoraService();
        while (Serial.available()) { char c = Serial.read(); if (c == '\r' || c == '\n') continue; if (!myHandleDebugChar(c)) cancel = true; }   // @commands, > and the page's D/d do not cancel
        delay(5);
      }
      if (cancel) Serial.println("Auto-start cancelled, the menu stays.");
      else { myIsSelected = true; myActionInfer(); return; }
    } else if (!myWeightsTrained) {
      Serial.println("No trained weights yet, so the menu stays. Collect images and train first.");
    }
  }
  myHandleMenuNavigation();
}



// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 1: IMAGE COLLECTION FUNCTIONS                                      ██
// ██                                                                          ██
// ██  DEPENDENCIES (functions called from Part 0):                            ██
// ██  - myResetMenuState()                     [Part 4]                       ██
// ██  - myReadTouch()                          [Part 4]                       ██
// ██                                                                          ██
// ██  VARIABLES USED (defined in Part 0):                                     ██
// ██  - myClassLabels[NUM_CLASSES], myThresholdPress, myLongPressTime                   ██
// ██  - u8g2 (OLED display object)                                            ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████


// ======================================================
// SHARED OLED RENDER HELPER
// Renders myRgbBuffer (must already be filled) to OLED.
// imageCount >= 0  -> show count badge (post-capture mode)
// imageCount == -1 -> show LIVE badge (preview mode)
// ======================================================
void myRenderRgbToOLED(int imageCount) {
  int myOledWidth  = u8g2.getDisplayWidth();
  int myOledHeight = u8g2.getDisplayHeight();
  int myScaleX = 240 / myOledWidth;
  int myScaleY = 240 / myOledHeight;

  u8g2.firstPage();
  do {
    for (int myOledX = 0; myOledX < myOledWidth; myOledX++) {
      for (int myOledY = 0; myOledY < myOledHeight; myOledY++) {
        size_t myPixelIndex = ((myOledY * myScaleY) * 240 + (myOledX * myScaleX)) * 3;
        uint8_t myBrightness = (myRgbBuffer[myPixelIndex]     +
                                myRgbBuffer[myPixelIndex + 1] +
                                myRgbBuffer[myPixelIndex + 2]) / 3;
        if (myBrightness > 100) u8g2.drawPixel(myOledX, myOledY);
      }
    }
    if (imageCount >= 0) {
      // Post-capture: count badge top-left
      u8g2.setFont(u8g2_font_ncenB10_tr);
      u8g2.setColorIndex(0);
      u8g2.drawBox(0, 0, 20, 15);
      u8g2.setColorIndex(1);
      u8g2.setCursor(3, 10);
      u8g2.print(String(imageCount));
    } else {
      // Live preview: LIVE badge top-right
      u8g2.setFont(u8g2_font_5x7_tf);
      u8g2.setColorIndex(0);
      u8g2.drawBox(50, 0, 22, 8);
      u8g2.setColorIndex(1);
      u8g2.drawStr(52, 7, "LIVE");
    }
  } while (u8g2.nextPage());
}

// Post-capture snapshot: convert fb -> myRgbBuffer then render with count badge
void myDisplayImageOnOLED(camera_fb_t* fb, int imageCount) {
  if (!myRgbBuffer) {
    Serial.println("RGB buffer not allocated - skipping OLED preview");
    return;
  }
  if (!fmt2rgb888(fb->buf, fb->len, fb->format, myRgbBuffer)) {
    Serial.println("Failed to convert JPEG to RGB888 for OLED");
    return;
  }
  myRenderRgbToOLED(imageCount);
}


void myActionCollect(int classIdx) {
  if (!mySDavailable) {
    Serial.println("No SD card - cannot collect images");
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "No SD card"); } while (u8g2.nextPage());
    delay(2000);
    myResetMenuState();
    return;
  }

  Serial.printf("\n>>> Collection mode: %s\n", myClassLabels[classIdx].c_str());
  Serial.println("Instructions:");
  Serial.println("  TAP (1-2 taps) = Capture image");
  Serial.println("  LONG PRESS (3+ taps) = Exit to menu");
  Serial.println("  Serial: 'T'=capture, 'L'=exit");
  
  myResetTouchState();  // Clear touch state when entering
  
  String path = "/images/" + myClassLabels[classIdx];
  if (!SD.exists("/images")) SD.mkdir("/images");
  if (!SD.exists(path)) SD.mkdir(path);


  // Count only the active class — no need to scan all folders on menu entry
  int counts[NUM_CLASSES] = {};
  File root = SD.open("/images/" + myClassLabels[classIdx]);
  if(root) {
    while(File file = root.openNextFile()) {
      if(!file.isDirectory() && (String(file.name()).endsWith(".jpg") || 
        String(file.name()).endsWith(".JPG"))) {
        counts[classIdx]++;
      }
      file.close();
    }
    root.close();
  }

  unsigned long lastCameraDrain = 0;  // how often we service the camera buffer
  unsigned long lastOLED = 0; unsigned long lastDebugPreview = 0;         // how often we actually update the OLED
  bool oledNeedsUpdate = false;
  bool shouldCapture = false;

  while (true) {
    unsigned long now = millis();

    // --- FAST LOOP: drain camera buffer every 50ms to prevent FB-OVF ---
    if (now - lastCameraDrain > 50) {
      lastCameraDrain = now;

      if (!shouldCapture) {  // don't grab preview frames if a capture is pending
        camera_fb_t* fb = esp_camera_fb_get();
        if (fb) {
          // slow live preview to the web page (only when it asked for debug frames)
          if (myDebugStream && now - lastDebugPreview > 1000) {
            lastDebugPreview = now;
            myDebugSendFrame('P', counts[classIdx], fb, -1, nullptr);
          }
          // Only pay for RGB conversion when the OLED is due for a refresh (250ms)
          if (now - lastOLED > 250 && myRgbBuffer) {
            if (fmt2rgb888(fb->buf, fb->len, fb->format, myRgbBuffer)) {
              oledNeedsUpdate = true;
              lastOLED = now;
            }
          }
          esp_camera_fb_return(fb);
        }
      }
    }

    // --- SLOW LOOP: render to OLED only when fresh RGB is ready ---
    if (oledNeedsUpdate) {
      oledNeedsUpdate = false;
      myRenderRgbToOLED(-1);  // -1 = show LIVE badge
    }

    // --- SERIAL INPUT ---
    if (Serial.available()) {
      char c = Serial.read();
      if (myHandleDebugChar(c)) {
        // handled: web page debug heartbeat
      } else if (c == 'l' || c == 'L') {
        myResetMenuState();
        return;
      } else if (c == 't' || c == 'T') {
        shouldCapture = true;
      }
    }

    // --- TOUCH INPUT - unified system ---
    int touchAction = myCheckTouchInput();
    if (touchAction == 2) {
      // Long press (3+ taps) - exit
      Serial.println("Exiting collection mode");
      myResetMenuState();
      return;
    } else if (touchAction == 1) {
      // Tap (1-2 taps) - capture
      shouldCapture = true;
    }

    // --- CAPTURE ---
    if (shouldCapture) {
      shouldCapture = false;
      camera_fb_t* fb = esp_camera_fb_get();
      if (fb) {
        String fileName = path + "/img_" + String(millis()) + ".jpg";
        File file = SD.open(fileName, FILE_WRITE);
        if (file) {
          file.write(fb->buf, fb->len);
          file.close();
          counts[classIdx]++;
          Serial.printf("Saved: %s (Total: %d)\n", fileName.c_str(), counts[classIdx]);
          myDisplayImageOnOLED(fb, counts[classIdx]);  // shows count badge
          myDebugSendFrame('C', counts[classIdx], fb, -1, nullptr);   // saved image to the web page
          delay(300);
          lastOLED = millis();  // don't immediately overwrite the snapshot with LIVE
        }
        esp_camera_fb_return(fb);
      }
    }

    delay(5);
  }
}

// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 2: TRAINING FUNCTIONS (FORWARD/BACKWARD PASS, OPTIMIZER)           ██
// ██                                                                          ██
// ██  DEPENDENCIES (functions called from Part 0):                            ██
// ██  - myAllocateMemory()                     [Part 0]                       ██
// ██  - myLoadWeights()                        [Part 0]                       ██
// ██  - mySaveWeights()                        [Part 0]                       ██
// ██  - myLoadImageFromFile()                  [Part 0]                       ██
// ██                                                                          ██
// ██  VARIABLES USED (defined in Part 0):                                     ██
// ██  - All neural network weight/gradient buffers                            ██
// ██  - myClassLabels[NUM_CLASSES], LEARNING_RATE, BATCH_SIZE, TARGET_EPOCHS            ██
// ██  - myTrainingData vector, myInputBuffer                                  ██
// ██  - u8g2 (OLED display object)                                            ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████


// ======================================================
// FORWARD PASS
// ======================================================
void myForwardPass(float* input, float* logits) {
  // Conv1: INPUT_SIZE x INPUT_SIZE x 3 -> CONV1_OUTPUT_SIZE x CONV1_OUTPUT_SIZE x CONV1_FILTERS
  for(int f=0; f<CONV1_FILTERS; f++) {
    int ob = f*CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE;
    for(int y=0; y<CONV1_OUTPUT_SIZE; y++) {
      for(int x=0; x<CONV1_OUTPUT_SIZE; x++) {
        float sum = 0;
        for(int ky=0; ky<3; ky++) {
          for(int kx=0; kx<3; kx++) {
            int inPos = ((y+ky)*INPUT_SIZE+(x+kx))*3;
            int wPos = f*27 + ky*9 + kx*3;
            sum += input[inPos]*myConv1_w[wPos] + 
                   input[inPos+1]*myConv1_w[wPos+1] + 
                   input[inPos+2]*myConv1_w[wPos+2];
          }
        }
        myConv1_output[ob + y*CONV1_OUTPUT_SIZE + x] = leaky_relu(clip_value(sum + myConv1_b[f]));
      }
    }
  }
  
  // Pool1: CONV1_OUTPUT_SIZE x CONV1_OUTPUT_SIZE -> POOL1_OUTPUT_SIZE x POOL1_OUTPUT_SIZE
  for(int f=0; f<CONV1_FILTERS; f++) {
    int ib=f*CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE, ob=f*POOL1_OUTPUT_SIZE*POOL1_OUTPUT_SIZE;
    for(int y=0; y<POOL1_OUTPUT_SIZE; y++) {
      for(int x=0; x<POOL1_OUTPUT_SIZE; x++) {
        int iy=y*2, ix=x*2;
        float maxVal = myConv1_output[ib + iy*CONV1_OUTPUT_SIZE + ix];
        maxVal = max(maxVal, myConv1_output[ib + iy*CONV1_OUTPUT_SIZE + ix+1]);
        maxVal = max(maxVal, myConv1_output[ib + (iy+1)*CONV1_OUTPUT_SIZE + ix]);
        maxVal = max(maxVal, myConv1_output[ib + (iy+1)*CONV1_OUTPUT_SIZE + ix+1]);
        myPool1_output[ob + y*POOL1_OUTPUT_SIZE + x] = maxVal;
      }
    }
  }
  
  // Conv2: POOL1_OUTPUT_SIZE x POOL1_OUTPUT_SIZE x CONV1_FILTERS -> CONV2_OUTPUT_SIZE x CONV2_OUTPUT_SIZE x CONV2_FILTERS
  for(int f=0; f<CONV2_FILTERS; f++) {
    int ob=f*CONV2_OUTPUT_SIZE*CONV2_OUTPUT_SIZE;
    for(int y=0; y<CONV2_OUTPUT_SIZE; y++) {
      for(int x=0; x<CONV2_OUTPUT_SIZE; x++) {
        float sum = 0;
        for(int c=0; c<CONV1_FILTERS; c++) {
          int ib=c*POOL1_OUTPUT_SIZE*POOL1_OUTPUT_SIZE;
          for(int ky=0; ky<3; ky++) {
            for(int kx=0; kx<3; kx++) {
              sum += myPool1_output[ib + (y+ky)*POOL1_OUTPUT_SIZE + (x+kx)] * 
                     myConv2_w[f*CONV2_IN_STRIDE + c*9 + ky*3 + kx];
            }
          }
        }
        myConv2_output[ob + y*CONV2_OUTPUT_SIZE + x] = leaky_relu(clip_value(sum + myConv2_b[f]));
      }
    }
  }
  
  // Dense layer
  for(int c=0; c<NUM_CLASSES; c++) {
    double sum = 0, comp = 0;
    for(int i=0; i<FLATTENED_SIZE; i++) {
      double term = myConv2_output[i] * myOutput_w[c*FLATTENED_SIZE + i];
      double y = term - comp;
      double t = sum + y;
      comp = (t - sum) - y;
      sum = t;
    }
    myDense_output[c] = clip_value((float)sum + myOutput_b[c], -50, 50);
  }
  
  // Softmax
  float mx = myDense_output[0];
  for(int i=1; i<NUM_CLASSES; i++) mx = max(mx, myDense_output[i]);
  float expSum = 0;
  for(int i=0; i<NUM_CLASSES; i++) expSum += exp(myDense_output[i]-mx);
  for(int i=0; i<NUM_CLASSES; i++) {
    logits[i] = myDense_output[i];
    myDense_output[i] = exp(myDense_output[i]-mx) / expSum;
  }
}

// ======================================================
// BACKWARD PASS
// ======================================================
void myBackwardDense(int label) {
  // myDense_grad is a per-image propagation signal — zero it fresh each image.
  // myOutput_w_grad and myOutput_b_grad use += so all images in the batch accumulate.
  // (Batch-level zeroing of those buffers is done once at the start of each batch loop.)
  memset(myDense_grad, 0, FLATTENED_SIZE * sizeof(float));
  for(int c=0; c<NUM_CLASSES; c++) {
    float error = myDense_output[c] - (c==label ? 1.0f : 0.0f);
    for(int i=0; i<FLATTENED_SIZE; i++) {
      myOutput_w_grad[c*FLATTENED_SIZE+i] += error * myConv2_output[i];  // += accumulates over the batch
      myDense_grad[i] += error * myOutput_w[c*FLATTENED_SIZE+i];
    }
    myOutput_b_grad[c] += error;  // += accumulates over the batch
  }
}

void myBackwardConv2() {
  for(int i=0; i<FLATTENED_SIZE; i++) {
    myConv2_grad[i] = myDense_grad[i] * leaky_relu_deriv(myConv2_output[i]);
  }
  
  // myConv2_w_grad and myConv2_b_grad are weight accumulators — do NOT zero
  // them here; the batch-level memset at the start of the batch loop handles that.
  // myPool1_grad IS zeroed here because it is a per-image propagation signal.
  memset(myPool1_grad, 0, POOL1_OUTPUT_SIZE*POOL1_OUTPUT_SIZE*CONV1_FILTERS*sizeof(float));
  
  for(int f=0; f<CONV2_FILTERS; f++) {
    int ob=f*CONV2_OUTPUT_SIZE*CONV2_OUTPUT_SIZE;
    for(int y=0; y<CONV2_OUTPUT_SIZE; y++) {
      for(int x=0; x<CONV2_OUTPUT_SIZE; x++) {
        float grad = myConv2_grad[ob+y*CONV2_OUTPUT_SIZE+x];
        myConv2_b_grad[f] += grad;
        for(int c=0; c<CONV1_FILTERS; c++) {
          int ib=c*POOL1_OUTPUT_SIZE*POOL1_OUTPUT_SIZE;
          for(int ky=0; ky<3; ky++) {
            for(int kx=0; kx<3; kx++) {
              int pi = ib+(y+ky)*POOL1_OUTPUT_SIZE+(x+kx);
              int wi = f*CONV2_IN_STRIDE+c*9+ky*3+kx;
              myConv2_w_grad[wi] += grad * myPool1_output[pi];
              myPool1_grad[pi] += grad * myConv2_w[wi];
            }
          }
        }
      }
    }
  }
}

void myBackwardPool1() {
  memset(myConv1_grad, 0, CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE*CONV1_FILTERS*sizeof(float));
  for(int f=0; f<CONV1_FILTERS; f++) {
    int ib=f*CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE, ob=f*POOL1_OUTPUT_SIZE*POOL1_OUTPUT_SIZE;
    for(int y=0; y<POOL1_OUTPUT_SIZE; y++) {
      for(int x=0; x<POOL1_OUTPUT_SIZE; x++) {
        int iy=y*2, ix=x*2;
        float poolVal = myPool1_output[ob+y*POOL1_OUTPUT_SIZE+x];
        float grad = myPool1_grad[ob+y*POOL1_OUTPUT_SIZE+x];
        if(myConv1_output[ib+iy*CONV1_OUTPUT_SIZE+ix] == poolVal) myConv1_grad[ib+iy*CONV1_OUTPUT_SIZE+ix] += grad;
        if(myConv1_output[ib+iy*CONV1_OUTPUT_SIZE+ix+1] == poolVal) myConv1_grad[ib+iy*CONV1_OUTPUT_SIZE+ix+1] += grad;
        if(myConv1_output[ib+(iy+1)*CONV1_OUTPUT_SIZE+ix] == poolVal) myConv1_grad[ib+(iy+1)*CONV1_OUTPUT_SIZE+ix] += grad;
        if(myConv1_output[ib+(iy+1)*CONV1_OUTPUT_SIZE+ix+1] == poolVal) myConv1_grad[ib+(iy+1)*CONV1_OUTPUT_SIZE+ix+1] += grad;
      }
    }
  }
}

void myBackwardConv1() {
  for(int i=0; i<CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE*CONV1_FILTERS; i++) {
    myConv1_grad[i] *= leaky_relu_deriv(myConv1_output[i]);
  }
  
  // myConv1_w_grad and myConv1_b_grad are weight accumulators — do NOT zero
  // them here; the batch-level memset at the start of the batch loop handles that.
  
  for(int f=0; f<CONV1_FILTERS; f++) {
    int ob=f*CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE;
    for(int y=0; y<CONV1_OUTPUT_SIZE; y++) {
      for(int x=0; x<CONV1_OUTPUT_SIZE; x++) {
        float grad = myConv1_grad[ob+y*CONV1_OUTPUT_SIZE+x];
        myConv1_b_grad[f] += grad;
        
        for(int ky=0; ky<3; ky++) {
          for(int kx=0; kx<3; kx++) {
            int inPos = ((y+ky)*INPUT_SIZE+(x+kx))*3;
            int wPos = f*27 + ky*9 + kx*3;
            myConv1_w_grad[wPos] += grad * myInputBuffer[inPos];
            myConv1_w_grad[wPos+1] += grad * myInputBuffer[inPos+1];
            myConv1_w_grad[wPos+2] += grad * myInputBuffer[inPos+2];
          }
        }
      }
    }
  }
}

// ======================================================
// OPTIMIZER
// ======================================================
void myAdamUpdate(float* w, float* g, float* m, float* v, int size, int step) {
  float b1=0.9f, b2=0.999f, eps=1e-6f;  // eps 1e-6f prevents float32 NaN
  float lr_t = LEARNING_RATE * sqrt(1-pow(b2,step)) / (1-pow(b1,step));
  for(int i=0; i<size; i++) {
    m[i] = b1*m[i] + (1-b1)*g[i];
    v[i] = b2*v[i] + (1-b2)*g[i]*g[i];
    w[i] -= lr_t*m[i]/(sqrt(v[i])+eps);
    w[i] = clip_value(w[i], -10, 10);
  }
}

void myUpdateWeights(int step) {
  myAdamUpdate(myConv1_w, myConv1_w_grad, myConv1_w_m, myConv1_w_v, CONV1_WEIGHTS, step);
  myAdamUpdate(myConv1_b, myConv1_b_grad, myConv1_b_m, myConv1_b_v, CONV1_FILTERS, step);
  myAdamUpdate(myConv2_w, myConv2_w_grad, myConv2_w_m, myConv2_w_v, CONV2_WEIGHTS, step);
  myAdamUpdate(myConv2_b, myConv2_b_grad, myConv2_b_m, myConv2_b_v, CONV2_FILTERS, step);
  myAdamUpdate(myOutput_w, myOutput_w_grad, myOutput_w_m, myOutput_w_v, OUTPUT_WEIGHTS, step);
  myAdamUpdate(myOutput_b, myOutput_b_grad, myOutput_b_m, myOutput_b_v, NUM_CLASSES, step);
}

// ======================================================
// TRAINING FUNCTION
// ======================================================


void myActionTrain() {
  if (!mySDavailable) {
    Serial.println("No SD card - cannot train");
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "No SD card"); } while (u8g2.nextPage());
    delay(2000);
    myResetMenuState();
    return;
  }

  Serial.println("\n>>> Training mode");
  Serial.println("Instructions:");
  Serial.println("  During training: 3+ taps = Save and exit");
  Serial.println("  After completion: TAP = Train again, 3+ taps = Exit");
  Serial.println("  Serial: 'T'=train again, 'L'=exit");
  
  myResetTouchState();  // Clear touch state when entering

  u8g2.firstPage();
  do {
    u8g2.setFont(u8g2_font_6x10_tf);
    u8g2.drawStr(0, 12, "TRAINING MODE");
    u8g2.drawStr(0, 24, "Loading...");
  } while (u8g2.nextPage());
  
  if (myLoadWeights()) {
    Serial.println("Continuing from saved weights");
  } else {
    //myAllocateMemory();
    Serial.println("Starting fresh training");
  }

  while (true) {
    // Load training data
    myTrainingData.clear();
    for(int i=0; i<NUM_CLASSES; i++) {
      File root = SD.open("/images/" + myClassLabels[i]);
      if (root) {
        while(File file = root.openNextFile()) {
          if(!file.isDirectory()) {
            String fn = String(file.name());
            if(fn.endsWith(".jpg") || fn.endsWith(".JPG")) {
              myTrainingData.push_back({file.path(), i});
            }
          }
          file.close();
        }
        root.close();
      }
    }
    
    if(myTrainingData.empty()) { 
      u8g2.firstPage();
      do { u8g2.drawStr(0, 20, "No Images!"); } while (u8g2.nextPage());
      delay(2000);
      myResetMenuState();
      return; 
    }

    // Sort by path for a deterministic split, then hold out the last
    // VALIDATION_IMAGES images per class as a validation set.
    std::sort(myTrainingData.begin(), myTrainingData.end(),
              [](const TrainingItem& a, const TrainingItem& b){ return a.path < b.path; });

    std::vector<TrainingItem> myValidationData;
    if (VALIDATION_IMAGES > 0) {
      int counts[NUM_CLASSES] = {};
      for (auto& item : myTrainingData) counts[item.label]++;
      int skip[NUM_CLASSES];
      for (int c = 0; c < NUM_CLASSES; c++) skip[c] = min(VALIDATION_IMAGES, counts[c]);

      std::vector<TrainingItem> trainOnly;
      int seen[NUM_CLASSES] = {};
      for (int i = (int)myTrainingData.size() - 1; i >= 0; i--) {
        int c = myTrainingData[i].label;
        if (seen[c] < skip[c]) {
          myValidationData.push_back(myTrainingData[i]);
          seen[c]++;
        } else {
          trainOnly.push_back(myTrainingData[i]);
        }
      }
      myTrainingData = trainOnly;
      Serial.printf("Val: %d images  Train: %d images\n",
                    (int)myValidationData.size(), (int)myTrainingData.size());
    }

    int total = myTrainingData.size();
    int batchesPerEpoch = (total + BATCH_SIZE - 1) / BATCH_SIZE;
    int totalBatches = TARGET_EPOCHS * batchesPerEpoch;
    
    Serial.printf("Training: %d images, %d batches\n", total, totalBatches);
    
    // Training loop
    std::vector<int> indices;
    for(int i=0; i<total; i++) indices.push_back(i);
    
    float runningLoss = 0;
    int lossCount = 0;
    
    for(int batch=0; batch<totalBatches; batch++) {
      // Check for exit during training
      if (Serial.available()) {
        char c = Serial.read();
        // 'l'/'L' or 'x'/'X' stops, like every other mode
        if (c == 'x' || c == 'X' || c == 'l' || c == 'L') {
          Serial.println("Stopping training...");
          mySaveWeights();
          myWeightsTrained = true; 
          myResetMenuState();
          return;
        }
      }
      
      // Touch input during training - check in background
      myCheckTouchBackground();  // Update touch state without blocking
      if (myPeekTouchAction() == 2) {
        myCheckTouchInput();  // Consume the action
        Serial.println("Long press - stopping training");
        mySaveWeights();
        myWeightsTrained = true; 
        myResetMenuState();
        return;
      }
      
      // Shuffle at epoch start
      if(batch % batchesPerEpoch == 0) {
        int epoch = batch/batchesPerEpoch + 1;
        Serial.printf("\n--- Epoch %d/%d ---\n", epoch, TARGET_EPOCHS);
        for(int i=total-1; i>0; i--) {
          int j = random(i+1);
          int tmp = indices[i];
          indices[i] = indices[j];
          indices[j] = tmp;
        }
      }
      
      int batchStart = (batch % batchesPerEpoch) * BATCH_SIZE;
      int batchEnd = min(batchStart + BATCH_SIZE, total);
      
      float batchLoss = 0;
      int correctCount = 0;
      
      // Zero ALL weight gradient buffers once per batch before accumulating
      // (zeroing inside the per-image functions would reset them between images).
      memset(myConv1_w_grad,  0, CONV1_WEIGHTS  * sizeof(float));
      memset(myConv1_b_grad,  0, CONV1_FILTERS  * sizeof(float));
      memset(myConv2_w_grad,  0, CONV2_WEIGHTS  * sizeof(float));
      memset(myConv2_b_grad,  0, CONV2_FILTERS  * sizeof(float));
      memset(myOutput_w_grad, 0, OUTPUT_WEIGHTS * sizeof(float));
      memset(myOutput_b_grad, 0, NUM_CLASSES    * sizeof(float));
      
      // Train on batch
      for(int i=batchStart; i<batchEnd; i++) {
        int idx = indices[i];
        TrainingItem& img = myTrainingData[idx];
        
        if(!myLoadImageFromFile(img.path.c_str(), myInputBuffer)) continue;
        
        float logits[NUM_CLASSES];
        myForwardPass(myInputBuffer, logits);
        
        float loss = -log(max(myDense_output[img.label], 1e-7f));
        batchLoss += loss;
        
        int pred = 0;
        for(int j=1; j<NUM_CLASSES; j++) if(myDense_output[j] > myDense_output[pred]) pred = j;
        if(pred == img.label) correctCount++;
        
        myBackwardDense(img.label);
        myBackwardConv2();
        myBackwardPool1();
        myBackwardConv1();
        
        // Update touch state during heavy computation
        // also peek for action here so a tap exits within one image,
        // not at the end of the entire batch (which can be a 5-15 second wait)
        if (i % 3 == 0) {
          myCheckTouchBackground();
          if (myPeekTouchAction() == 2) {
            myCheckTouchInput();  // consume
            Serial.println("Long press - stopping training");
            mySaveWeights();
            myWeightsTrained = true; 
            myResetMenuState();
            return;
          }
          if (Serial.available()) {
            char c = Serial.read();
            if (c == 'x' || c == 'X' || c == 'l' || c == 'L') {
              Serial.println("Stopping training...");
              mySaveWeights();
              myWeightsTrained = true; 
              myResetMenuState();
              return;
            }
          }
        }
      }
      
      myUpdateWeights(batch+1);
      
      float avgLoss = batchLoss / (batchEnd - batchStart);
      float batchAcc = (float)correctCount / (batchEnd - batchStart);
      runningLoss += avgLoss;
      lossCount++;
      
      // Update display
      if((batch+1) % 5 == 0) {
        float displayLoss = runningLoss / lossCount;
        u8g2.firstPage();
        do {
          u8g2.setFont(u8g2_font_5x7_tf);   // _6x10_tf
          u8g2.setCursor(0, 12); u8g2.print("Training...");
          u8g2.setCursor(0, 24); 
          u8g2.print("B:"); u8g2.print(batch+1); 
          u8g2.print("/"); u8g2.print(totalBatches);
          u8g2.setCursor(0, 36); 
          u8g2.print("L:"); u8g2.print(displayLoss, 3);
          u8g2.print(" A:"); u8g2.print((int)(batchAcc*100)); u8g2.print("%");
        } while (u8g2.nextPage());
        runningLoss = 0;
        lossCount = 0;
      }
      
      if((batch+1) % 10 == 0) {
        Serial.printf("Batch %d/%d - Loss: %.4f - Acc: %.1f%%\n", 
                     batch+1, totalBatches, avgLoss, batchAcc*100);
      }
    }
    
    Serial.println("\n--- Training Complete ---");

    // Run forward pass on held-out validation images and report accuracy.
    if (!myValidationData.empty()) {
      int valCorrect = 0;
      int valCount   = 0;
      for (auto& vitem : myValidationData) {
        if (!myLoadImageFromFile(vitem.path.c_str(), myInputBuffer)) continue;
        float logits[NUM_CLASSES];
        myForwardPass(myInputBuffer, logits);
        int pred = 0;
        for (int j = 1; j < NUM_CLASSES; j++)
          if (myDense_output[j] > myDense_output[pred]) pred = j;
        if (pred == vitem.label) valCorrect++;
        valCount++;
      }
      if (valCount > 0) {
        Serial.printf("Validation Accuracy: %.1f%%  (%d/%d correct)\n",
                      100.0f * valCorrect / valCount, valCorrect, valCount);
      }
    }

    mySaveWeights();
    myWeightsTrained = true;

    u8g2.firstPage();
    do { 
      u8g2.drawStr(0, 12, "DONE!");
      u8g2.drawStr(0, 24, "Tap:Again");
      u8g2.drawStr(0, 36, "3+Taps:Exit");
    } while (u8g2.nextPage());

    myResetTouchState();
    
    Serial.println("Waiting for input...");
    Serial.println("  Serial: T=train again  L=exit");
    Serial.println("  Touch:  1-2 taps=train again  3+taps=exit");
    while (true) {
      if (Serial.available()) {
        char c = Serial.read();
        // 'l'/'L' or 'x'/'X' leaves, like every other mode
        if (c == 'x' || c == 'X' || c == 'l' || c == 'L') {
          myResetMenuState();
          return;
        } else if (c == 't' || c == 'T') {
          break;
        }
      }
      int touchAction = myCheckTouchInput();
      if (touchAction == 2) {
        myResetMenuState();
        return;
      } else if (touchAction == 1) {
        Serial.println("Starting new training cycle");
        break;
      }
      delay(10);
    }
  }
}


// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 3: INFERENCE FUNCTION - OPTIMIZED                                  ██
// ██                                                                          ██
// ██  DEPENDENCIES (functions called from Part 0):                            ██
// ██  - myLoadWeights()    // Weights loaded in setup()                       ██
// ██  - myForwardPass()                        [Part 2]                       ██
// ██  - myRgbBuffer (global, allocated in setup)                              ██
// ██                                                                          ██
// ██  VARIABLES USED (defined in Part 0):                                     ██
// ██  - myInputBuffer, myDense_output (probabilities)                         ██
// ██  - myClassLabels[NUM_CLASSES], myThresholdPress                                    ██
// ██  - u8g2 (OLED display object)                                            ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████


void myActionInfer() {
  // Guard: refuse to run if no trained weights are loaded
  if (!myWeightsTrained) {
    Serial.println("ERROR: No trained weights! Please run menu item 4 (Train) first.");
    u8g2.firstPage();
    do {
      u8g2.setFont(u8g2_font_6x10_tf);
      u8g2.drawStr(0, 12, "No weights!");
      u8g2.drawStr(0, 24, "Train first");
      u8g2.drawStr(0, 36, "(menu item 4)");
    } while (u8g2.nextPage());
    delay(3000);
    myResetMenuState();
    return;
  }
  Serial.println("\n>>> Inference mode - OPTIMIZED");
  Serial.println("Instructions:");
  Serial.println("  T or L exit to menu");

  
  myResetTouchState();  // Clear touch state when entering
  
  // Weights already loaded in setup() - just verify PSRAM is ready
  if (!myInputBuffer || !myDense_output) {
    Serial.println("ERROR: Memory not allocated - cannot infer");
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "NOT READY!"); } while (u8g2.nextPage());
    delay(2000);
    myResetMenuState();
    return;
  }
  
  // Pre-compute resize lookup tables (done once)
  static int sy_lookup[INPUT_SIZE];
  static int sx_lookup[INPUT_SIZE];
  static bool lookup_initialized = false;
  
  if (!lookup_initialized) {
    for(int i=0; i<INPUT_SIZE; i++) {
      sy_lookup[i] = min((int)((i+0.5)*240.0/INPUT_SIZE), 239);
      sx_lookup[i] = min((int)((i+0.5)*240.0/INPUT_SIZE), 239);
    }
    lookup_initialized = true;
    Serial.println("Resize lookup tables initialized");
  }
  
  // Timing arrays for 10-frame batches
  unsigned long frameTimes[10];
  int frameIndex = 0;
  int pred = 0;  // Store prediction outside loop for printing
  unsigned long myInferCount = 0;   // frame number for the debug frames
  
  while (true) {
    unsigned long frameStart = millis();
    myLoraService();   // receive, report, transmit
    
    // Serial input check (fast, every frame)
    if (Serial.available()) {
      char c = Serial.read();
      if (myHandleDebugChar(c)) {
        // handled: web page debug heartbeat
      } else if (c == 't' || c == 'T' || c == 'l' || c == 'L') {
        myResetMenuState();
        return;
      }
    }
    
    // Get camera frame
    camera_fb_t * fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera frame failed - retrying");
      delay(10);
      continue;
    }
    
    // Check if RGB buffer is allocated
    if (!myRgbBuffer) {
      Serial.println("ERROR: myRgbBuffer not allocated!");
      esp_camera_fb_return(fb);
      delay(10);
      continue;
    }
    
    // Convert JPEG to RGB (reusing pre-allocated buffer)
    if (fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, myRgbBuffer)) {
      
      // Optimized resize using lookup tables
      for(int y=0; y<INPUT_SIZE; y++) {
        int sy = sy_lookup[y];
        int sy_offset = sy * 240;
        int dst_y_offset = y * INPUT_SIZE;
        
        for(int x=0; x<INPUT_SIZE; x++) {
          int srcIdx = (sy_offset + sx_lookup[x]) * 3;
          int dstIdx = (dst_y_offset + x) * 3;
          myInputBuffer[dstIdx] = myRgbBuffer[srcIdx] * 0.003921569f;      // /255.0
          myInputBuffer[dstIdx+1] = myRgbBuffer[srcIdx+1] * 0.003921569f;
          myInputBuffer[dstIdx+2] = myRgbBuffer[srcIdx+2] * 0.003921569f;
        }
      }
      
      // Run inference
      float myLogits[NUM_CLASSES];
      myForwardPass(myInputBuffer, myLogits);
      
      myInferCount++;
      // Find prediction
      pred = 0;
      for(int i=1; i<NUM_CLASSES; i++) {
        if(myDense_output[i] > myDense_output[pred]) pred = i;
      }

      myLoraCount(pred, myDense_output[pred]);   // count this frame for the next report

      // Every 10th frame: draw live image + label overlay on OLED.
      // Done HERE while myRgbBuffer is still valid (before fb is returned).
      if (frameIndex == 9) {
        myDebugSendFrame('I', (int)myInferCount, fb, pred, myLogits);   // every 10th inference to the web page
        int oW = u8g2.getDisplayWidth();
        int oH = u8g2.getDisplayHeight();
        int scX = 240 / oW;
        int scY = 240 / oH;
        u8g2.firstPage();
        do {
          // Draw downsampled camera image
          for (int ox = 0; ox < oW; ox++) {
            for (int oy = 0; oy < oH; oy++) {
              int pi = ((oy * scY) * 240 + (ox * scX)) * 3;
              uint8_t bright = (myRgbBuffer[pi] + myRgbBuffer[pi+1] + myRgbBuffer[pi+2]) / 3;
              if (bright > 100) u8g2.drawPixel(ox, oy);
            }
          }
          // Label overlay bar at bottom
          u8g2.setFont(u8g2_font_5x7_tf);
          u8g2.setColorIndex(0);
          u8g2.drawBox(0, oH - 9, oW, 9);
          u8g2.setColorIndex(1);
          char buf[20];
          snprintf(buf, sizeof(buf), "%s %d%%",
                   myClassLabels[pred].c_str(),
                   (int)(myDense_output[pred] * 100));
          u8g2.drawStr(1, oH - 1, buf);
        } while (u8g2.nextPage());
      }
    }
    
    esp_camera_fb_return(fb);
    
    // Record frame timing
    frameTimes[frameIndex] = millis() - frameStart;
    float fps2 = 1000.0 / frameTimes[frameIndex];
    bool myShowLine = (myPrintEvery <= 1) || (myInferCount % myPrintEvery == 0);
    if (myShowLine) Serial.printf("Frame %d: %lu ms (%.1f FPS) ", frameIndex+1, frameTimes[frameIndex], fps2);
    frameIndex++;
    if (myShowLine) {
      Serial.printf("Current Pred: %s (%.1f%%) | All:", 
                     myClassLabels[pred].c_str(), myDense_output[pred]*100);
      for(int i=0; i<NUM_CLASSES; i++) Serial.printf(" %.0f%%", myDense_output[i]*100);
      Serial.println();
    }
   
    // Every 10th frame: touch exit check (OLED image already drawn above before fb return)
    if (frameIndex >= 10) {
      int touchVal = myReadTouch();
      if (myTouchExit && touchVal > myThresholdPress) {
        Serial.println("Touch detected - exiting inference");
        delay(200);
        myResetMenuState();
        return;
      }
      frameIndex = 0;
    }
  }
}


// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 4: MENU SYSTEM FUNCTIONS                                           ██
// ██                                                                          ██
// ██  DEPENDENCIES (functions called from Part 0):                            ██
// ██  - myActionCollect(int classIdx)          [Part 1]                       ██
// ██  - myActionTrain()                        [Part 2]                       ██
// ██  - myActionInfer()                        [Part 3]                       ██
// ██                                                                          ██
// ██  VARIABLES USED (defined in Part 0):                                     ██
// ██  - myClassLabels[NUM_CLASSES]                                                      ██
// ██  - myTotalItems, myThresholdPress, myThresholdRelease                    ██
// ██  - myLastActivityTime, myLastTapTime, myTapCooldown                      ██
// ██  - myIsTouching, myLongPressTriggered, myMenuIndex, myIsSelected         ██
// ██  - u8g2 (OLED display object)                                            ██
// ██                                                                          ██
// ██  NOTE: This part is called from loop() in Part 0                         ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████


// LoRa messages screen: shows the last message sent and heard. Send with >text, leave with t, l, @menu or a long touch.
void myLoraDrawChat() {
  char b[20];
  u8g2.firstPage();
  do {
    u8g2.setFont(u8g2_font_5x7_tf);
    snprintf(b, sizeof(b), "LoRa ch%d T%lu R%lu", myLoraChannel, myLoraTxN, myLoraRxN);
    u8g2.drawStr(0, 7, b);
    for (int k = 0; k < 2; k++) {
      const char* m = k == 0 ? myLoraLastTx : myLoraLastRx;
      const char* tag = k == 0 ? "TX " : "RX ";
      char l1[20] = "", l2[20] = "";
      if (m[0]) {
        const char* sp = strchr(m, ' ');
        snprintf(l1, sizeof(l1), "%s%.*s", tag, sp ? (int)(sp - m) : 10, m);
        if (sp) snprintf(l2, sizeof(l2), "%.14s", sp + 1);
      } else snprintf(l1, sizeof(l1), "%s-", tag);
      u8g2.drawStr(0, 15 + k * 16, l1);
      u8g2.drawStr(0, 23 + k * 16, l2);
    }
  } while (u8g2.nextPage());
}

void myActionLoraChat() {
  Serial.println(F("\n=== LORA MESSAGES ==="));
  Serial.printf("board %s, channel %d (%.1f MHz), radio %s\n", myLoraName, myLoraChannel, myLoraFreq(), myLoraOk ? "ok" : "OFF");
  Serial.println(F("Send:   >your text   then Enter.   Messages from other boards appear here.   Leave:  " MY_LEAVE_HINT));
  myResetTouchState();
  myWantMenu = false;
  myLoraDrawChat();
  unsigned long lastDraw = millis();
  while (true) {
    myLoraService();
    while (Serial.available()) {
      char c = Serial.read();
      if (myHandleDebugChar(c)) { myLoraDrawChat(); lastDraw = millis(); continue; }   // >text, @commands and the page's D/d
      if (c == 't' || c == 'T' || c == 'l' || c == 'L') myWantMenu = true;
    }
    if (myWantMenu || myCheckTouchInput() == 2) { myWantMenu = false; Serial.println(F("Leaving LoRa messages")); myResetMenuState(); return; }
    if (millis() - lastDraw > 1000) { myLoraDrawChat(); lastDraw = millis(); }
    delay(5);
  }
}

String myMenuLabel(int i) {
  if (i <= NUM_CLASSES)     return myClassLabels[i - 1];
  if (i == NUM_CLASSES + 1) return "Train";
  if (i == NUM_CLASSES + 2) return "Infer";
  return "LoRa msg";
}

void myResetMenuState() {
  myIsSelected = false;
  myResetTouchState();  // Use unified touch reset
  myLastActivityTime = millis();
  myDrawMenu();
}

void myDrawMenu() {
  // ===== SERIAL MENU =====
  Serial.println("\n=== MENU ===");
  for (int i = 1; i <= myTotalItems; i++) {
    String label = myMenuLabel(i);

    if (i == myMenuIndex) Serial.print(" > ");
    else                 Serial.print("   ");

    Serial.printf("%d. %s\n", i, label.c_str());
  }
  Serial.println("Menu: t=next  l=select  or press an item's digit   >text = send a LoRa message   @help = all commands");

  // ===== OLED MENU =====
  u8g2.firstPage();
  do {
    u8g2.setFont(u8g2_font_6x10_tf);
#if MY_TOUCH_ENABLED
    u8g2.drawStr(0, 8, "TAP:Next HOLD:Ok");
#else
    u8g2.drawStr(0, 8, "t=next l=ok");
#endif

    int myStartItem = (myMenuIndex <= NUM_CLASSES) ? 1 : myMenuIndex - 2;

    for (int i = 0; i < 3; i++) {
      int cur = myStartItem + i;
      if (cur > myTotalItems) break;

      String label = myMenuLabel(cur);

      int y = 18 + i * 9;
      if (cur == myMenuIndex)
        u8g2.drawStr(0, y, ("> " + label).c_str());
      else
        u8g2.drawStr(0, y, ("  " + label).c_str());
    }
  } while (u8g2.nextPage());
}

// Helper: execute the currently selected menu item
void myExecuteMenuItem(int idx) {
  if (idx <= NUM_CLASSES)        myActionCollect(idx - 1);
  else if (idx == NUM_CLASSES+1) myActionTrain();
  else if (idx == NUM_CLASSES+2) myActionInfer();
  else                           myActionLoraChat();
}

void myHandleMenuNavigation() {
  unsigned long myCurrentMillis = millis();

  // --------------------------------------------------------------------------
  // SERIAL INPUT
  // --------------------------------------------------------------------------
  if (!myIsSelected && Serial.available()) {
    char c = Serial.read();

    // Single-digit direct selection (works for NUM_CLASSES up to 9+2=11 items via digit keys)
    if (myHandleDebugChar(c)) {
      // handled: web page debug heartbeat
    }
    else if (c >= '1' && c <= '9') {
      int newIndex = c - '0';
      if (newIndex <= myTotalItems) {
        myMenuIndex = newIndex;
        myIsSelected = true;
        myLastActivityTime = myCurrentMillis;
        myExecuteMenuItem(myMenuIndex);
      }
    }
    else if (c == 't' || c == 'T') {
      if (myCurrentMillis - myLastTapTime > myTapCooldown) {
        myMenuIndex++;
        if (myMenuIndex > myTotalItems) myMenuIndex = 1;
        myDrawMenu();
        myLastTapTime = myCurrentMillis;
        myLastActivityTime = myCurrentMillis;
      }
    }
    else if (c == 'l' || c == 'L') {
      myIsSelected = true;
      myLastActivityTime = myCurrentMillis;
      myExecuteMenuItem(myMenuIndex);
    }
  }

  // --------------------------------------------------------------------------
  // TOUCH INPUT
  // --------------------------------------------------------------------------
  if (!myIsSelected) {
    int touchAction = myCheckTouchInput();
    
    if (touchAction == 1) {
      // Tap detected - advance menu
      if (myCurrentMillis - myLastTapTime > myTapCooldown) {
        myMenuIndex++;
        if (myMenuIndex > myTotalItems) myMenuIndex = 1;
        myDrawMenu();
        myLastTapTime = myCurrentMillis;
        myLastActivityTime = myCurrentMillis;
      }
    }
    else if (touchAction == 2) {
      // Long press detected - select menu item
      myIsSelected = true;
      myLastActivityTime = myCurrentMillis;
      myExecuteMenuItem(myMenuIndex);
    }
  }
}
