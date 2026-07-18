# Prepoznavanje europskih glavnih gradova iz fotografija

Ovaj repozitorij sadrži početnu implementaciju projekta završnog rada čiji je cilj razvoj modela dubokog učenja za prepoznavanje europskih glavnih gradova na temelju fotografija njihovih urbanih središta.

## Cilj projekta

Cilj projekta je izraditi skup podataka s fotografijama europskih glavnih gradova te trenirati model koji će na temelju nove fotografije pokušati odrediti kojem gradu ona pripada.

Projekt se trenutačno nalazi u fazi provjere izvedivosti i automatiziranog prikupljanja podataka.

## Prikupljanje podataka

Fotografije se prikupljaju putem Mapillary API-ja korištenjem Python skripte.

Kako bi prikupljanje bilo što standardiziranije, za svaki grad koriste se:

* jednaki radijus oko središnje koordinate grada
* jednaki razmak prostorne mreže
* jednaka veličina područja pretraživanja
* ograničen broj fotografija iz iste Mapillary sekvence
* filtriranje panoramskih i fisheye fotografija
* odabir fotografija koje su prostorno raspoređene unutar centra grada

Na taj se način izbjegava ručno biranje znamenitosti ili subjektivno određivanje zanimljivih lokacija.

## Trenutačno testirani gradovi

Skripta je uspješno testirana za:

* Zagreb
* Madrid
* Rim
* Beč

Prikupljene fotografije ne nalaze se u ovom repozitoriju zbog veličine skupa podataka i uvjeta korištenja izvora fotografija.

## Korištene tehnologije

* Python
* Mapillary API
* Requests
* python-dotenv
* CSV za spremanje metapodataka

Za treniranje modela planira se korištenje biblioteke TensorFlow ili PyTorch te pristupa transfer learninga.

## Struktura projekta

```text
project/
├── collect_city_images.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

Lokalno se nakon pokretanja skripte stvara mapa:

```text
dataset/
├── Zagreb/
├── Madrid/
├── Rome/
└── Vienna/
```

Mapa `dataset` nije uključena u Git repozitorij.

## Instalacija

Potrebno je imati instaliran Python.

Instalacija potrebnih paketa:

```bash
python -m pip install -r requirements.txt
```

## Mapillary API token

Potrebno je napraviti `.env` datoteku u glavnoj mapi projekta:

```env
MAPILLARY_ACCESS_TOKEN=OVDJE_UPISATI_TOKEN
```

API token ne smije se objaviti na GitHubu.

Primjer datoteke može se spremiti kao `.env.example`:

```env
MAPILLARY_ACCESS_TOKEN=
```

## Pokretanje

Postavke grada definiraju se u Python skripti:

```python
CITY_NAME = "Zagreb"
CENTER_LAT = 45.8131
CENTER_LON = 15.9775
```

Pokretanje skripte:

```bash
python collect_city_images.py
```

Skripta sprema fotografije i pripadajuće metapodatke u mapu `dataset`.

## Planirani nastavak rada

Sljedeći koraci uključuju:

1. proširenje skupa podataka na veći broj europskih glavnih gradova
2. dodatnu provjeru kvalitete i uklanjanje vrlo sličnih fotografija
3. podjelu podataka na skupove za treniranje, validaciju i testiranje
4. treniranje više modela za klasifikaciju slika
5. usporedbu rezultata različitih arhitektura
6. analizu pogrešno klasificiranih gradova

## Predloženi naziv završnog rada

**Razvoj i evaluacija modela dubokog učenja za prepoznavanje europskih glavnih gradova na temelju fotografija urbanih središta**
