# boiler

## ~~Craving~~Crawling for Coubs

I used a slightly patched version of https://github.com/flute/coub-crawler to download almost 24 hours of coubs in one run.

I've discovered that downloaded audios are truncated by the video length
however it's common that coub audio tracks are longer than video clips themselves.
I'm going to ignore that issue for now.

I used ffmpeg to convert mp4s to wavs:

```bash
parallel -j6 -n1  ffmpeg -nostdin -i {} -vn -ar 16000 -ac 1 wav/{/.}.wav ::: video/*.mp4
```

Based on the [distribution](https://github.com/glamp/bashplotlib) of audio lengths I've decided to pad each audio clip to 9s by repeating
shorter clips and truncating longer ones:

```console
proger@rt:~/coub-crawler/monthlyLog$ soxi -D wav/*.wav | hist -b 20 -p 🍄

 5644|           🍄
 5347|           🍄
 5050|           🍄
 4753|           🍄
 4456|           🍄
 4159|           🍄
 3862|           🍄
 3565|           🍄
 3268|           🍄
 2971|           🍄
 2674|           🍄
 2377|           🍄
 2080|           🍄
 1783|           🍄
 1486|           🍄
 1189|           🍄
  892|           🍄
  595|         🍄🍄🍄
  298|       🍄🍄🍄🍄🍄
    1| 🍄🍄🍄🍄🍄🍄🍄🍄🍄🍄🍄🍄🍄🍄🍄🍄🍄🍄🍄🍄🍄

----------------------------------
|            Summary             |
----------------------------------
|       observations: 9344       |
|      min value: 0.162562       |
|        mean : 9.168637         |
|      max value: 20.247812      |
----------------------------------
```