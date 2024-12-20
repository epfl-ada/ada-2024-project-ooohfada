<div style="border: 2px solid red; padding: 10px; display: inline-block; border-radius: 15px"><p>
  <h2> Suffering from a bad buzz ? Here is where it's at...</h2>
  <br />

  Let’s face it — you wouldn’t still be reading this unless you’d royally messed up, would you? You’ve stepped on the internet landmine, and now your channel is caught in the crosshairs of a… <b>BAD BUZZ</b> 😱. <br />
  <br />

  If you’ve scrolled through the previous sections and still can’t find a fix, it’s because your situation requires more than the usual “post more consistently” or “engage with your audience” advice. But fear not! While the internet’s collective fury can feel like a raging storm, we’ve armed ourselves with data analysis and research to help you turn this ship around. <br />
  <br />

  <h3>The Challenges of a Big YouTuber in a Crisis</h3>
  <br />

  From now on, our analysis focuses on the top <b>4%</b> (2955 out of 74788) of channels (that have had suffered from a decline) — the ones with more than 1 million subscribers. These creators are not only the most visible but also the most vulnerable to public backlash. <br />
  <br />

  When a big YouTuber faces a crisis, the stakes are high. The potential loss of subscribers, sponsors, and revenue can be catastrophic. But the good news is that are also the ones with the most influence and thus power in their upload content, so recovery is possible! <br />
  <br />

  Our analysis of these big YouTubers (think >1M subscribers — we feel flattered that you're here) has revealed some fascinating trends. Whether it’s a Twitter faux pas, a controversial statement, or a scandal you never saw coming, the strategies for clawing back your reputation vary dramatically depending on the moves you make right now.(Insert dramatic pause for effect. You’re feeling the pressure, aren’t you?) <br />
  <br />

  But hey, before you run off to film a tearful apology, let’s talk about why some YouTubers manage to recover from their digital nightmares while others end up in the dreaded “where are they now?” category. Spoiler: it’s not all about talent or luck. <br />
  <br />

  <h3>The Recovery Rates of Big YouTubers</h3>
  <br />

  We’re not going to leave you hanging with just data and no solutions. Let’s reclaim that subscribe button together.<br />
 <br />
  For this, we needed to analyze the recovery rates of these big YouTubers based on the type of video they uploaded. First thing you need to know is that not all of the big youtubers have recovered from their bad buzz efficiently. <br />

  <div class="flourish-embed flourish-election" data-src="visualisation/20860562"><script src="https://public.flourish.studio/resources/embed.js"></script><noscript><img src="https://public.flourish.studio/visualisation/20860562/thumbnail" width="100%" alt="election visualization" /></noscript></div>

  One could think that that’s not good news, but you have to see the glass as half full. What it actually shows is that when focusing only on big YouTubers, the recovery rate is not better... BUT also NOT WORSE !<br />
  <br />

  As before, let's know dive into the details. The following plot shows the recovery rates of these big YouTubers based on the category of their channel: <br />

  <div id="flourish-container">
    <div id="flourish-BY" class="flourish-embed flourish-election" data-src="visualisation/20860572"></div>
    <div id="flourish-ALL" class="flourish-embed flourish-election" data-src="visualisation/20801957" style="display: none;"></div>
  </div>

  <button id="toggle-button" onclick="toggleFlourish()">Click for a reminder: All Channels</button>

  <style>
    #toggle-button {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      font-size: 16px;
      padding: 12px 16px;
      background-color: #ffffff;
      color: #333333;
      border-radius: 8px;
      border: none;
      box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
      cursor: pointer;
      transition: all 0.3s ease;
    }

    #toggle-button:hover {
      background-color: #f8f8f8;
    }

    #toggle-button:focus {
      outline: none;
      box-shadow: 0 0 8px rgba(0, 123, 255, 0.5); /* Soft blue glow */
    }
  </style>

  <script src="https://public.flourish.studio/resources/embed.js"></script>
  <script>
    let currentFlourish = 'BY'; // Default plot is 'BY'
    function toggleFlourish() {
      const button = document.getElementById('toggle-button');
      const flourishBY = document.getElementById('flourish-BY');
      const flourishALL = document.getElementById('flourish-ALL');
      if (currentFlourish === 'ALL') {
        // Switch to big YouTubers plot
        flourishALL.style.display = 'none';
        flourishBY.style.display = 'block';
        button.textContent = 'Click for a reminder: All Channels';
        currentFlourish = 'BY';
      } else {
        // Switch to all channels plot
        flourishBY.style.display = 'none';
        flourishALL.style.display = 'block';
        button.textContent = 'Click to focus on Big YouTubers';
        currentFlourish = 'ALL';
      }
    }
  </script>
  <br>
  <br>
  <div style="display: flex; align-items: center;">
    <img src="/assets/img/star.png" alt="Note on the plot" style="width: 20px; margin-left: 20px;">
    <p style="flex: 1; margin-left: 20px;">
      While the amount of samples is not the same, there is still some interesting conclusion to make from the comparision of the recovery rates of big youtubers and all channels.
    </p>
  </div>

  <div style="display: flex; align-items: center;">
    <img src="/assets/img/podium.png" alt="Description of image" style="width: 150px;">
    <p style="flex: 1; margin-left: 20px;">
      <b>This time, the winners are ...</b>
      At the very top of the podium, the category <span style="color: #004AAD;">"Travel & Events"</span> confirms its stronger than the average forgivability  but this with a shoking <b>100%</b> recovery rate, followed by <span style="color: #004AAD;">"Autos & Vehicles"</span> with a solid <b>90%</b> - that's good news if you make Road Trips videos ;) 
      <br>
      Of course, there is also a <b>big loser</b>... the <span style="color: #b51a00;">"Music"</span> category! Again, music seems to be a non-forgiving place, with a very modest recovery rate of <b>16.85%</b> (two times lower than with all channels).
      <br>
    </p>
  </div>

  <div style="display: flex; align-items: center;">
    <p style="flex: 1; margin-right: 20px;">
      There are also some categories where big youtubers manage to <b>positively reverse the trend</b> and recover better than the average for all channels in this category. It is the case for <span style="color: #004AAD;">"Autos & Vehicles"</span>, <span style="color: #004AAD;">"Comedy"</span>, <span style="color: #004AAD;">"Film and Animation"</span>, <span style="color: #004AAD;">"People & Blogs"</span>, <span style="color: #004AAD;">"Science & Technology"</span> and our loved <span style="color: #004AAD;">"Travel & Events"</span>. <br>
      But if there are positive reverses, there also <b>negative</b> ones... <span style="color: #b51a00;">"Music"</span> and <span style="color: #b51a00;">"News & Politics"</span> are the two categories where big youtubers recover worse than the average for all channels in this category. Not good news if you are in one of them...<br />
      (Honorable mention to "Nonprofits & Activism" category that sadly doesn't contain a single big youtuber...) 
    </p>
    <img src="/assets/img/switch.png" alt="Note on the plot" style="width: 150px;">
  </div>

  While those findings are already interesting, you should be questionning yourself about what makes these difference that big. No need to worry, we have the answer for you !<br />
  <br />


  <h3>The Secret Sauce? Your Titles</h3>
  <br />

  Yes, you read that right. <b>Titles</b>. Those few dozen characters can either save your channel or sink it further. Think about it: what’s the first thing a viewer sees before they even decide to click? Your title! And if you’ve learned anything from this wild ride of YouTube, it’s that first impressions are everything. <br />
  <br />

  Now, here’s where the science kicks in. You’ve probably heard of <b>LLMs (Large Language Models)</b>, right? They’re super-smart AI tools that can analyse and recognise patterns in text faster than you can say, “Oops, I tweeted what?!” <br />
  <br />

  <div style="display: flex; align-items: center;">
    <img src="/assets/img/lama.png" alt="Note on the plot" style="width: 100px;">
    <p style="flex: 1; margin-left: 15px;">
      We used this tech wizardry to dive into the aftermath of bad buzz for dozens of big YouTubers. By leveraging the <b>Mistral model</b> from the <a href="https://ollama.com/library/mistral" target="_blank"><b>OLLAMA</b></a> open source project, we analysed the types of videos uploaded immediately following their PR disasters.
    </p>
  </div>
  
  <b>What did we look for ?</b> Patterns. Specifically, whether these videos fell into one (or more) of these categories:

  <ol>
      <li> Apology videos: The classic “I’m sorry” trope—effective or overdone? </li>
      <li> Addressing the decline: Are you facing the issue head-on or pretending nothing’s wrong?</li>
      <li> Comeback announcements: Bold, confident, and ready to win back hearts.</li>
      <li> Break announcements: Sometimes, stepping back is stepping forward.</li>
      <li> Collaboration videos: Is strength in numbers the way to go?</li>
      <li> Clickbait videos: Risky, but can it work in your favor?</li>
  </ol>

  <br />
  And guess what? Our findings are as wild as the comment section on your latest video !<br />
  <br />

  The LLM analyzed cautiously the title of all the <b>107,618</b> videos uploaded by these big youtubers during their respective crisis. The following plot shows how many videos fall into each of these categories among this considerable amount of videos, according to the LLM, based on their title.<br />
  <br />

  [TODO: plot(apology=198,clickbait=78351,break=568,comeback=902,featuring=4150,decline_addressed=2675)]<br />
  <br />

  We can already see that the most common strategy (maybe because it's the easiest ?) is to spam clickbait videos. But is it the most effective strategies ? The response lies only in the data...<br />
  <br />

  <div style="border-left: 4px solid #51247a; padding-left: 20px; font-size: 18px; background-color: #DFC5FE">
  Again, in order to get some unbiased results and make the proportion of recoveries meaningful, we performed <b>propensity score matching</b> on our decline dataset for big youtubers to observe the impact of each type of video. This is mandatory since the amount of channels uploading videos of these types is nothing compared to those you don't.
  </div>
  <br />

  And here is what you could not wait to see...<br />
  <br />

  TODO:PLOT PSMATCHING WHEN CHANGED (WAITING FOR FEU VERT DE NATHAN)
  <br />

  <br />
  This is a bit disappointing we know... clickbait videos is the only type of videos that seems to have a positive impact on the recovery rate of big youtubers, that's not what your heart tells you to do. But don't worry, these results have to be observed considering that they might suffer from a baseline bias. For example, one could think that the apology video that you surely want to make for the sake of your conscience would have a terrible impact on your recovery chances, but it could be because the channels that make apology videos are the ones that are in the worst situation. <br />
  <br />
  
  And maybe it's not the only type of video your heart (or brain!) tells you to make. Maybe you want to make a clickbait video, but also an apology video, and a featuring video, and a comeback video, and ... BUT you don't know if it's a good idea. We got you! Just read the next section to know more about the strategies to adopt !<br />

  <br />
  <h3>The (Exclusive) Strategy to Adopt</h3>
  <br />
  Let's first define what we mean by <b>"strategy"</b>. In our case, a strategy is a combination of the different types of videos that a big youtuber can upload during a crisis. For example, you could decide to upload at least one apology video, at least one clickbait video and at least one collaboration video (note that one video could be an apology, clickbait and a featuring at the same time, because why not ?); and you don't close the door to the other types of videos (among the our 6 video types of interest), you are just not sure yet. This would be considered as the strategy "Apology + Clickbait + Featuring". <br />
  <br />

  Secondly, we define an <b>"exclusive strategy"</b> as a strategy where you decide to upload video(s) of certan types and to NOT upload any other type (among the our 6 video types of interest) because you are sure you don’t want to stoop to that. For our previous example (the strategies adopted or Apology, Clickbait and Featuring), this would be considered as the <b>EXCLUSIVE</b> strategy "Apology + Clickbait + Featuring". <br />
  <br />

  Let's now see which (exclusive) strategies are the more adopted by the big youtubers, and which ones are the most efficient. For this, we've gathered the proportion of youtubers that have adopted each (exclusive) strategy, and the associated recovery rate. All you have to do is to click on the circle(s) of the strategy you want to adopt. If you are more into an exclusive strategy, just switch to the second graph, and select your exclusive strategy exactly the same way.<br />
  <br />

  Select a graph and enjoy:

  <select id="graphSelector" onchange="updateData()" style = "margin-left: 40px;">
    <option value="graph1">Strategies Proportion and Recovery Rate</option>
    <option value="graph2">Exclusive Strategies Proportion and Recovery Rate</option>
  </select>

  <style>
    #graphSelector {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      font-size: 16px;
      padding: 12px 16px;
      background-color: #ffffff;
      color: #333333;
      border-radius: 8px;
      appearance: none; /* Remove default dropdown arrow */
      box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
      transition: all 0.3s ease;
      width: auto;
    }

    #graphSelector:hover {
      background-color: #f8f8f8;
    }

    #graphSelector:focus {
      outline: none;
      box-shadow: 0 0 8px rgba(0, 123, 255, 0.5); /* Soft blue glow */
    }

    #graphSelector option {
      padding: 10px;
      background-color: #ffffff;
      color: #333333;
      transition: background-color 0.2s;
    }

    #graphSelector option:hover {
      background-color: #007bff;
      color: white;
    }
  </style>

  <div style="text-align: center;"> 
      <svg id="graph"></svg>
      <div id="legend" style="margin-top: -250px; font-size: 12px;">
          <div style="display: flex; justify-content: center; align-items: center; gap: 20px;">
              <div style="display: flex; align-items: center;">
                  <svg width="20" height="20"><circle cx="10" cy="10" r="8" fill="#004AAD"></circle></svg>
                  <span style="margin-left: 5px;">Strategy Not Applied</span>
              </div>
              <div style="display: flex; align-items: center;">
                  <svg width="20" height="20"><circle cx="10" cy="10" r="8" fill="#b51a00"></circle></svg>
                  <span style="margin-left: 5px;">Strategy Applied</span>
              </div>
              <div style="display: flex; align-items: center;">
                  <svg width="20" height="20"><line x1="0" y1="10" x2="20" y2="10" stroke="rgba(97, 23, 125, 0.4)" stroke-width="2"></line></svg>
                  <span style="margin-left: 5px;">Strategies Applied Together</span>
              </div>
          </div>
      </div>
  </div>

  <svg id="graph" width="400" height="10"></svg>

  <script src="https://d3js.org/d3.v7.min.js"></script>

  <script>
    const svg = d3.select("#graph")
      .attr("viewBox", "-250 -180 500 500");

    // Remove any existing script tags for the graphs
    function removeScript(src) {
      const script = document.querySelector(`script[src="${src}"]`);
      if (script) script.remove();
    }

    // Load a script dynamically
    function loadScript(src, callback) {
      const script = document.createElement("script");
      script.src = src;
      script.onload = callback;
      document.head.appendChild(script);
    }

    // Update the graph based on dropdown selection
    function updateData() {
      const selectedGraph = document.getElementById("graphSelector").value;

      // Clear the current graph
      svg.selectAll("*").remove();

      // Remove any previously loaded scripts
      removeScript("graph1.js");
      removeScript("graph2.js");

      // Load the selected graph's script
      if (selectedGraph === "graph1") {
        loadScript("assets/js/graph1.js", function() {
          drawGraph1(svg); // Call the drawing function for graph1
        });
      } else {
        loadScript("assets/js/graph2.js", function() {
          drawGraph2(svg); // Call the drawing function for graph2
        });
      }
    }

    // Initial load
    updateData();
  </script>

  <br /> 
  <br /> 
  The graph is designed so that you can play around with the different strategies and see how they would impact your recovery rate. Have some fun with it, experiment with the combinations, until you find <b>YOUR</b> gem. <br />
  <br />

  While the proportions for each non-exclusive strategy is not insightful, the recovery tells us a lot about the the path to follow... Here are our recommendations:<br />
  <br />

  <ul>
    <li><b>The "GO!"s:</b> The most efficient (and widely adopted) strategies are the ones that include the <b>clickbait</b> videos with a recovery rate of almost <b>50%</b>. Combining this and a video that addresses your bad buzz seems to be the best duo if you wanted to start with two differents strategy. <br /><br /></li>
    <li><b>The "NO!"s:</b> Apologizing in a video definitely seems to be counterproductive (recovery rate of 37%). All the worst recovery rates contain video(s) of this type. Even if your better judgment tells you to do it, it might be better to avoid starting by beging for forgiveness. <br /></li>
  </ul>

  When it comes to exclusive strategies- if you are sure about the type of video you do <b>NOT</b> want to upload, here are some insights:
  <br /><br />
  <ul>
    <li><b>The case of Clickbait:</b> A key observation on the proportions is that the <b>majority</b> of big youtubers did not limit themselves to only clickbait videos, the proportion falls down from 76% to 31% when we make the strategy exclusive. However, it keeps the highest recover rate (52.2%) among the other exclusive single-type strategies. <br /><br /></li>
    <li><b>The troubled waters:</b> On the other hand, we quickly notice that a lot of exclusive combinations have never been tested by our top 4% youtubers. This is the case for the <b>"Break + Comeback"</b> strategy, that which yet appears to be a logical one. Go on if you really want to risk loose your channel trying one of these never explored strategies, but that is definitely not what we would recommend. <br /><br /></li>
    <li><b>Saving your apologies by clickbaiting:</b> A major point is that when combined with <b>"Clickbait"</b> (only), the <b>"Apology"</b> video comes out of nowhere to demonstrate the fabulous recovery rate of <b>100%</b>. But don't get too much enthusiastic about this result, the proportion of youtubers that have adopted this strategy is very lo -it is still encouraging if that was the perfect plan in your mind. <br /><br /></li>
    <li><b>Multi-type strategies:</b> The only multi-type strategies that have already been widely adopted are the <b>"Clickbait + Featuring"</b> (<b>12%</b>  of the big youtubers tried this) and the <b>"Clickbait + Bad Buzz Addressed" (5%)</b> strategies. And while the first one seems not to be optimal (only 49% of them recovered), the <b>"Clickbait + Bad Buzz Addressed"</b> demonstrates a recovery rate of 54%, giving you more than a 50% chance of saving your channel<br /></li>
  </ul>

  <br />

  <br />

  </p>
</div>