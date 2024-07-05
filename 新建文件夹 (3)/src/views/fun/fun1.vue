<template>
  <div>
    <h1>功能点 1</h1>
    <video ref="videoPlayer" width="780" height="480" controls :paused="isPaused">
      <source src="1211055185-1-192.mp4" type="video/mp4">
      您的浏览器不支持视频标签。
    </video>
    <button @click="togglePlay">播放/暂停</button>
  </div>
</template>

<script>
export default {
  name: 'Fun1',
  data() {
    return {
      isPaused: true
    }
  },
  methods: {
    togglePlay() {
      this.isPaused = !this.isPaused;
      if (this.$refs.videoPlayer) {
        if (this.isPaused) {
          this.$refs.videoPlayer.pause();
        } else {
          this.$refs.videoPlayer.play();
        }
      }
    }
  }
}
</script>
