# ระดับสัญญาณ analog, HV ของ PMT และช่วง glue — ข้อกำหนดและตัวเลขจากข้อมูลจริง

เอกสารนี้รวบรวมที่มาของคำแนะนำว่า **ยอดสัญญาณ analog ไม่ควรเกินครึ่งหนึ่งของ input range
(= 250 mV เมื่อใช้ range 500 mV)** พร้อมตัวเลขที่วัดได้จากระบบของเรา และผลที่จะเกิดขึ้นถ้าลดอัตราขยายลง

เขียนเมื่อ 2026-09-16 จากข้อมูลคืนวันที่ 8, 9, 14 และ 15 กันยายน 2026

---

## 1. ข้อกำหนดจากคู่มือ

### 1.1 ยอดสัญญาณไม่เกินครึ่งหนึ่งของ input range
**Licel PM-HV Photomultiplier Module R9880U Manual ข้อ 5.2 (Peak Signal)**

> "The peak signal can significantly exceed the background level as long as the integral current
> is below 100 µA. The analog signal should fit into the input range of the preamplifier and the
> ADC. Due to the fluctuations of the lidar signal the following rule of thumb will keep the
> signal inside the ADC range."
>
> "**The peak signal should not exceed half of the input range.** For a 100mV input range this
> corresponds to 50mV."

ระบบเราตั้ง input range ไว้ที่ 500 mV (ค่าสูงสุดของ TR40) ตามกฎนี้ยอดสัญญาณจึงไม่ควรเกิน **250 mV**

เหตุผลที่คู่มือเผื่อไว้ครึ่งหนึ่ง: สัญญาณ lidar แกว่งระหว่างช็อต ค่าเฉลี่ยที่ยังไม่ถึงเพดาน
ก็มีช็อตเดี่ยวที่ชนเพดานได้

### 1.2 สิ่งที่เกิดขึ้นเมื่อชนเพดาน
**Licel TR Manual (หัวข้อโครงสร้างข้อมูล analog)**

> "The analog data has a additional flag indicating that the sum incorporates a overflow value.
> If for one bin the ADC gives either 0 or 0xFFF the flags is set for the sum and indicates that
> either an over- or underflow has occured at this special bin. **The sum at these points may not
> correspond to the physical mean value** as the actual ADC value could have been -10 or above 4095."

- ค่าใน bin ที่ชนเพดานไม่ใช่ค่าจริงและกู้คืนไม่ได้ คู่มือทำได้แค่ติด flag ไว้
- **Licel Ethernet Controller Manual ข้อ 9.2.3** อธิบายว่า flag เป็น bitmask ต่อ dataset (bit0 = TR0, bit1 = TR1)
- library ของ Licel (`licel_data.py`) ก็อ่าน flag นี้ออกมาเฉย ๆ ไม่มีสูตรแก้ค่า

### 1.3 กระแส anode เฉลี่ยห้ามเกิน 100 µA
**PMT manual ข้อ 5.1 (Maximum DC Signal)**

> "The maximum allowed average anode current for a R7400 or R9880 PMT is 100 µA. This corresponds
> to **5mV in the analog mode at a 50 Ω input** averaged over a 100 ms time. **This should never be
> exceeded.** The Licel PMT has a high current clamp down which will reduce the signal once more
> than 120 µA are drawn from the tube for more than 100...1000ms. If this happens the signal will
> be severely distorted and not useful for any LIDAR measurement."

**Licel PMT datasheet** ตรงกันเรื่องกระแสเฉลี่ยสูงสุด 0.1 mA (= 100 µA) และระบุ gain 2×10⁵–2×10⁶, ช่วง HV −100 V ถึง −1 kV, ความกว้าง pulse ของโฟตอนเดี่ยวน้อยกว่า 2 ns **แต่ระบุจุดที่วงจรป้องกันเริ่มทำงานต่างจากคู่มือ:** datasheet บอกเกิน 0.5 mA นาน 1 วินาที ส่วนคู่มือบอกเกิน 120 µA นาน 100–1000 ms ยังไม่ทราบว่าโมดูลของเราใช้เกณฑ์ไหน (รายละเอียดและผลที่วัดได้ช่วงกลางวันอยู่ในเอกสารแยกเรื่องกระแส PMT)

### 1.4 คำเตือนเรื่องการลด HV
**PMT manual ข้อ 5.1 และข้อ 3**

> "Reducing the high voltage below 800V might help on first glance but is not optimal as a long
> term solution. The PMT cathode can provide a limited amount of photons over its lifetime,
> exposing the tube to a constant DC light will shorten this lifetime."

> "The specified gain reduction of 10E-3 is achieved when using high voltage settings between
> 800V and 1000V."

**ระบบเราใช้ HV 750 V ซึ่งต่ำกว่าช่วง 800–1000 V ที่คู่มืออ้างถึงอยู่แล้ว** การลด HV ลงอีก
จึงยิ่งออกนอกช่วงที่ผู้ผลิตระบุ และมีผลข้างเคียงกับ dead time (ข้อ 4.2)

### 1.5 ช่วงที่ analog กับ photon ใช้ร่วมกันได้
**Licel Ethernet Controller Manual ข้อ 9.7.1**

> "The main idea of the signal combination is that there is a region where both signals are valid
> and have a high signal to noise ratio. For typical Mini-PMT that region extends from 0.5 to
> 10 MHz in the photon counting."

ของเราวัดเองได้ 10–40 MHz ประเด็นสำคัญคือ **ต้องมี bin ที่ตกอยู่ในช่วงนี้จริง**
ไม่เช่นนั้นจะหาค่า glue ของ profile นั้นไม่ได้

---

## 2. สิ่งที่วัดได้จากระบบของเรา

### 2.1 ยอดสัญญาณ analog (กลางคืน, ช่อง ∥)

| ไฟล์ | ยอด analog | ตำแหน่ง | bin ที่ชนเพดาน | ชนถึงระยะ |
|---|---|---|---|---|
| 8 ก.ย. 21:00 | 492.5 mV | 38 m | 23 | 889 m |
| 8 ก.ย. 23:00 | 449.3 mV | 38 m | 27 | 1429 m |
| 15 ก.ย. 18:30 | 494.3 mV | 98 m | 66 | 3368 m |
| 15 ก.ย. 22:00 | 494.3 mV | 98 m | 33 | 1125 m |

- **เกินขีด 250 mV ประมาณ 2 เท่า** และ 33 จาก 37 profile ของวันที่ 15 ติด flag นี้
- ช่อง ⊥ ยอดอยู่ที่ 26–86 mV ผ่านเกณฑ์สบาย
- ในเมฆหนายอดจะชนยาวขึ้น วันที่ 14–15 ชนถึง 2.5–3.4 km

### 2.2 อัตรานับ photon ของช่อง ∥ (ค่าดิบ, กลางคืน)

| ระยะ | 8 ก.ย. 21:00 | 15 ก.ย. 18:30 |
|---|---|---|
| 150 m | 197.9 MHz | 201.8 MHz |
| 1 km | 149.7 MHz | 174.6 MHz |
| 2 km | 66.6 MHz | 105.2 MHz |
| 3 km | 30.1 MHz | 61.5 MHz |
| 5 km | 9.0 MHz | 17.1 MHz |

ช่วง glue 10–40 MHz จึงตกที่ระยะ 3–5 km ซึ่ง analog เหลือน้อยมาก และถ้ามีเมฆบังก่อนถึงระยะนั้น
จะไม่มี bin ให้ fit เลย นี่คือสาเหตุที่ช่อง ∥ ต้องใช้ค่า glue คงที่แทบทุกคืน

### 2.3 ผลที่ตามมาในการประมวลผล
- bin ที่ชนเพดานถูกตัดออกจากการหาค่า glue และติด flag ใน sheet `Saturation_par` (ค่า 2 = ไม่มีค่าที่ใช้ได้)
- หลังเมฆที่ทำให้ analog ล้น สัญญาณ analog ยังเพี้ยนต่ออีกราว 1 km (สูงเกินจริง 30–100 %) แล้วตกต่ำกว่าเส้นฐาน
- ค่า glue ของช่อง ∥ จึงมาจากค่าคงที่ 87 MHz/mV ไม่ได้มาจาก profile นั้นเอง

---

## 3. ถ้าลดอัตราขยายลงครึ่งหนึ่งจะได้อะไร

| หัวข้อ | ตอนนี้ | ถ้าลดลง 2 เท่า |
|---|---|---|
| ยอด analog ช่อง ∥ | 450–494 mV (ชนเพดาน) | ~225–250 mV ผ่านเกณฑ์ข้อ 5.2 |
| อัตรานับที่ 2 km | 67–105 MHz | 33–53 MHz เข้าใกล้ช่วง glue |
| อัตรานับที่ 3 km | 30–62 MHz | 15–31 MHz **อยู่ในช่วง glue 10–40 MHz** |
| bin สำหรับหาค่า glue | มักไม่มีเลย | มีทุก profile ที่เมฆไม่บังก่อน 3 km |
| SNR ที่ระยะไกล | อ้างอิง | แย่ลงราว 1.4 เท่า (จำนวนนับลดครึ่ง) |
| ช่อง ⊥ | 26–86 mV | 13–43 mV ยังใช้ได้ แต่ SNR แย่ลงเท่ากัน |

**สรุป:** ลดอัตราขยาย 2 เท่าแก้ได้ทั้งเรื่องยอดชนเพดานและเรื่องหาค่า glue แลกกับ SNR
ที่ระยะไกลราว 1.4 เท่า ถ้าลดมากกว่านี้ SNR จะแย่ลงโดยไม่ได้ประโยชน์เพิ่ม

---

## 4. ทางเลือกในการลดอัตราขยาย

### 4.1 ใส่ ND filter หน้าช่อง ∥ (แนะนำ)
- ลดทั้งสัญญาณและแสงพื้นหลังเท่ากัน จึงช่วยเรื่องกระแส anode ตอนกลางวันด้วย (ข้อ 1.3)
- ไม่กระทบ dead time และไม่กระทบตำแหน่ง valley ของ discriminator
- ใส่เฉพาะช่อง ∥ ได้ ไม่ต้องแตะช่อง ⊥ ที่ปกติดีอยู่แล้ว
- ND 0.3 ≈ ลดลง 2 เท่า
- **ข้อควรระวัง:** อัตราส่วน ⊥/∥ เปลี่ยน ต้องหาค่า calibration ของ δ ใหม่

### 4.2 ลด HV ของช่อง ∥
- คู่มือเตือนว่าการลด HV ต่ำกว่า 800 V ไม่ใช่ทางแก้ระยะยาว และเราอยู่ที่ 750 V แล้ว
- pulse ของโฟตอนจะเตี้ยลงเทียบกับ discriminator ที่ตั้งไว้ระดับ 8 ทำให้
  - จำนวนนับหายไปบางส่วน
  - **dead time เปลี่ยน** ซึ่งเราเพิ่งวัดได้ 4.8 ns ที่ HV 750 V ต้องวัดใหม่
- ถ้าจะทำ ควรวัด pulse height distribution ใหม่เพื่อหาตำแหน่ง valley ของ discriminator ด้วย

### 4.3 เปลี่ยน input range ของ TR
- ทำไม่ได้ เพราะ 500 mV เป็นค่าสูงสุดของ TR40 อยู่แล้ว

### 4.4 ไม่แก้อะไร
- โปรแกรมรับมือได้ระดับหนึ่ง: ตัด bin ที่ชนเพดาน ใช้ photon แทนในช่วงหลังเมฆ และใช้ค่า glue คงที่
- แต่ค่า glue ของช่อง ∥ จะไม่ได้มาจากข้อมูลคืนนั้น และ δ ต้องพึ่งค่าคงที่ต่อไป

---

## 5. ขั้นตอนตรวจสอบหลังปรับ

1. อัดไฟล์มืดก่อน: ปิดหน้าเลเซอร์และหน้ากล้อง เปิด pretrigger อย่างน้อย 2401 shots ทั้งสองช่อง
2. อัด profile กลางคืนที่ฟ้าใส 3–5 profile
3. รัน Step 2 แล้วดูใน sheet `QC_calibration`
   - `par_analog_peak_mv` ควรต่ำกว่า 250
   - `par_analog_peak_over_half_range` ควรเป็น 0
   - `par_glue_fit_mode` ควรเป็น `robust_median_gain` ไม่ใช่ `fixed_gain_fallback`
   - `par_solar_dc_mv` ตอนกลางวันควรต่ำกว่า 5
4. ถ้าเปลี่ยน HV ต้องวัด dead time ใหม่ และวัดช่วง toggle ใหม่
5. ตั้งค่า Fallback glue gain ใหม่ตามค่าที่วัดได้
6. หาค่า calibration ของ δ ใหม่ เพราะอัตราส่วนระหว่างช่องเปลี่ยน

---

## 6. ข้อจำกัดของเอกสารนี้
- ตัวเลข "ลด 2 เท่า" คำนวณจากยอดที่ชนเพดานอยู่แล้ว ค่าจริงก่อนชนอาจสูงกว่า 494 mV จึงอาจต้องลดมากกว่า 2 เท่า
- ความสัมพันธ์ระหว่าง HV กับ gain ของ R9880U ไม่มีเป็นสมการในเอกสารที่เรามี datasheet ให้แค่ช่วง gain
  2×10⁵–2×10⁶ ตลอดช่วง HV จึงประเมินไม่ได้ว่าต้องลด HV กี่โวลต์ ต้องวัดจริง
- ผลกระทบต่อ SNR คำนวณจากสถิติการนับ (√N) ยังไม่ได้ทดสอบกับข้อมูลจริงหลังปรับ

---

## เอกสารอ้างอิง
1. Licel PM-HV Photomultiplier Module R9880U Manual — ข้อ 3, 4, 5.1, 5.2, 5.6, 5.7
2. Licel Ethernet Controller Manual — ข้อ 9.2.3 (overflow dataset), ข้อ 9.7 (gluing, dead time)
3. Licel TR Manual (HS version) — โครงสร้างข้อมูล analog และ clip flag
4. Licel Photomultiplier module datasheet (06/14)
5. `Licel_TCPIP_Python/Licel/licel_data.py` — การอ่าน clipping information
6. ข้อมูลการวัดของเรา: `Optimization LIDAR/raw-data-08-09-2026`, `raw-data09-09-2026`,
   `raw-data-14-09-2026`, `raw-data-15-09-2026` (sheet QC_calibration ของ Step 2)
